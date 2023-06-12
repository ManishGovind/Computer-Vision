import sys
import torch
from math import gcd
import functools
from einops import rearrange

from MyNet import MyNet
from MyNet2 import MyNet2
from SelfAttention import SelfAttention
from MHSelfAttention import MHSelfAttention
from TransformerBlock import TransformerBlock
from Encoder import Encoder

def lcm(*numbers):  # least common multiple
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def main():
    t1 = (4,8,5)  # produces 120
    print(*t1)
    res = lcm(*t1)
    print(res)
    dim = -3
    pad_offset = (0,) * (-1 - dim) * 2
    print(pad_offset)

    at = torch.tensor([[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]])
    bt = torch.tensor([[1, 2, 1, 1],
        [3, 4, 2, 5],
        [1, 3, 6, 7],
        [1, 4, 6, 8]])
    print('------at------')
    print(at)
    print('------bt------')
    print(bt)
    ct = at * bt  # does element by element multiplication
    print(ct)

    ct2 = torch.matmul(at,bt) # matrix multiplication
    print(ct2)

    ct2a = torch.einsum('ij, ij -> ij', at, bt) # element by element mult.
    print(ct2a)

    ct3 = torch.einsum('ij, jk -> ik', at, bt) # matrix mult.
    print(ct3)
    #a1 = torch.tensor([[1,2],[3,4]])
    #b1 = torch.tensor([[10,20],[30,40]])
    #ct3x = torch.einsum('ij, xy -> jx', a1, b1) # matrix mult.
    #print(ct3x)
    #ct3y = torch.einsum('ij, km -> im', at, bt) # matrix mult.
    #print(ct3y)

    #ct3a = torch.einsum('ij, jk -> i', at, bt) # add cols after matrix mult.
    #print(ct3a) # produces 4 result
    #d2 = ct3a.unsqueeze(dim=1) # add dim to make it 4x1, dim=0 makes it 1x4
    #print(d2.shape)

    #ct4 = torch.einsum("ii -> i", at) # diagonal elements
    #print(ct4)
    #ct5 = torch.einsum("ii -> ", at) # sum diagonal elements - trace
    #print(ct5)
    #ct6 = torch.einsum("ij -> j", at) # sum column elements (row wise sum)
    #print(ct6)
    #ct7 = torch.einsum('ij, ij -> ij', at, bt) # element wise product
    #print(ct7)

    #ct8 = torch.einsum('ij, ij, ij -> ij', at, at, at) # cube elements
    #print(ct8)

    #ct9 = torch.einsum('ij -> ji', at)  # transpose
    #print(ct9)

    #d1 = torch.tensor([3, 5, 7, 9])
    #d2 = torch.tensor([1, 2, 3, 4])
    #douter = torch.einsum('i, j -> ij', d1, d2) # outer product
    #print(douter)
    #douter2 = torch.einsum('i, j -> i', d1, d2) # reduction after outer product
    #print(douter2)
    #dinner = torch.einsum('i, i -> ', d1, d2) # inner product
    #print(dinner)

    #dfrobenius = torch.einsum("ij, ij -> ", at, at) # frobenius norm
    ## sum of squares of all elements of a matrix
    #print('Frobenius norm...')
    #print(dfrobenius)

    #batch_tensor_1 = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    #print(batch_tensor_1)
    #batch_tensor_2 = torch.arange(2 * 4 * 3).reshape(2, 3, 4)
    #print(batch_tensor_2)
    #dmul = torch.einsum('bij, bjk -> bik', batch_tensor_1, batch_tensor_2) # batch matrix multiplication
    #print(dmul)

    #dt = torch.randn((3,5,4,6,8,2,7,9)) # 8 dimensions
    #print(dt.shape)
    #esum = torch.einsum("ijklmnop -> p", dt) 
    ## marginalize or sum over dim p
    #print(esum) # produces 9 numbers, try op instead of p

    ##---------rearrange tensors using einsum-------------
    #x = torch.randn((1024,256))
    #xt1 = rearrange(x, 'n d -> () n d') # (1,1024,256)
    #print(xt1.shape)
    #xt2 = rearrange(x, '(h w) d -> h w d',h=8) # (8,128,256)
    #print(xt2.shape)

    #xt3 = rearrange(xt2, 'b n (h d) -> (b h) n d', h = 8)
    #print(xt3.shape) # (64,128,32)

    ##xt4 = rearrange(xt2, 'b (w n) d -> b w n d', n = 16) # (8,8,16,256)
    #xt4b = rearrange(xt2, 'b (w n) d -> b w d n', n = 16) # (8,8,256,16)
    ##print(xt4b.shape) 
    #net = MyNet()
    ##y = net(xt4) # will give an error, as net expects input of 16
    #y = net(xt4b) # (8,8,256,4)
    #print(y.shape)

    #a = torch.tensor([[2,3,4],[5,6,7]])
    #az = rearrange(a,'i j->j i') # does transpose
    #print(az)

    #xd = rearrange(xt4b,'b n s d-> b n d s')  # for compressing the segment 256 into 32
    #print(xd.shape)
    #net2 = MyNet2()  # 256 to 32 network
    #yy = net2(xd)
    #print(yy.shape)
    #yy2 = rearrange(yy,'b n d c-> b n c d') # assume 16 is embedding
    #print(yy2.shape)  # (8,8,32,16)

    #x = torch.randn((8,128,64))
    ## ---test self attention
    #sattn = SelfAttention(64)
    #out = sattn(x)
    #print('----self attention output shape-----')
    #print(out.shape)

    ## ---test multi headed self attention
    #x = torch.randn((8,128,512))
    #mhsa = MHSelfAttention(512)
    #out = mhsa(x)
    #print('----mhsa output shape-----')
    #print(out.shape)

    ## ---test transformer block
    #x = torch.randn((8,128,512))
    #mhsa = TransformerBlock(512)
    #out = mhsa(x)
    #print('----transformer block output shape-----')
    #print(out.shape)

    ## ---test encoder layers
    #x = torch.randn((8,128,512))
    #enc = Encoder(512)
    #out = enc(x)
    #print('----encoder layers output shape-----')
    #print(out.shape)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
