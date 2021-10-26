# the code is modified from https://github.com/d-li14/involution/blob/main/cls/mmcls/models/utils/involution_cuda.py
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template
import math

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_shift_kernel = kernel_loop + '''
extern "C"
__global__ void shift_forward_kernel(
const ${Dtype}* bottom_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    const int g = c / ${group};
    const int s = - (g - (${shift} / 2));
    ${Dtype} value = 0;
    if (${dim} == 2){
        if ((h + s >= 0 && h + s< ${height}) &&
            (w >= 0 && w < ${width})) {
             const int offset = ((n * ${channels} + c) * ${height} + h + s) * ${width} + w;
             value = bottom_data[offset];
        }
    } else {
        if ((h >= 0 && h < ${height}) &&
            (w + s >= 0 && w + s< ${width})) {
            const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w + s;
            value = bottom_data[offset];
            }
    }
    top_data[index] = value;
  }
}
'''


_shift_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void shift_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${height} / ${width};
    const int c = (index / ${height} / ${width}) % ${channels};
    const int h = (index / ${width}) % ${height};
    const int w = index % ${width};
    const int g = c / ${group};
    const int s = - (g - (${shift} / 2));
    ${Dtype} value = 0;
    if (${dim} == 2){
        if ((h - s >= 0 && h - s< ${height}) &&
            (w >= 0 && w < ${width})) {
             const int offset = ((n * ${channels} + c) * ${height} + h - s) * ${width} + w;
             value = top_diff[offset];
        }
    } else {
        if ((h >= 0 && h < ${height}) &&
            (w - s >= 0 && w - s < ${width})) {
            const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w - s;
            value = top_diff[offset];
        }
    }
    bottom_diff[index] = value;
  }
}
'''


class _shift(Function):
    @staticmethod
    def forward(ctx, input, shift, dim):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = input.new(batch_size, channels, height, width)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('shift_forward_kernel', _shift_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, 
                            height=height, width=width,
                            shift=shift, dim=dim, group=int(math.ceil(channels/shift))
                            )

            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input)
        ctx.shift, ctx.dim = shift, dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input = ctx.saved_tensors[0]
        shift, dim = ctx.shift, ctx.dim
        batch_size, channels, height, width = input.size()

        grad_input = None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   height=height, width=width,
                   shift=shift, dim=dim, group=int(math.ceil(channels/shift))
              )

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('shift_backward_grad_input_kernel',
                                _shift_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, None, None
 
def _shift_cuda(input, shift, dim):
    """ involution kernel
    """
    assert shift >=3 and shift % 2 == 1
    assert dim == 2 or dim == 3

    if input.is_cuda:
        out = _shift.apply(input, shift, dim)
    else:
        raise NotImplementedError
    return out


class Shift(nn.Module):
    def __init__(self,
                 kernel_size,
                 dim):
        super(Shift, self).__init__()
        self.kernel_size = kernel_size
        self.dim = dim
        assert dim == 2 or dim == 3
        assert kernel_size % 2 == 1

    def forward(self, x):
        if self.kernel_size == 1:
            return x

        out = _shift_cuda(x, self.kernel_size, self.dim)
        return out


def torch_shift(x, shift_size, dim):
    B_, C, H, W = x.shape
    pad = shift_size // 2

    x = F.pad(x, (pad, pad, pad, pad) , "constant", 0)
    xs = torch.chunk(x, shift_size, 1)
    x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-pad, pad+1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, pad, H)
    x_cat = torch.narrow(x_cat, 3, pad, W)
    return x_cat