print("Loading Module...")
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pandas as pd
import os
if os.system("cl.exe"):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

print("Loading Function...")
def initfilter(num_filters):
    for i in range(0,num_filters,1):
        reg=np.random.rand(3,3).astype(np.float32)
        np.savetxt('ai_save_model/filter{}.txt'.format(i), reg ,fmt='%.0f')

def pathfrom(path):
    list1=os.listdir(path)
    list2=os.listdir(path)
    for i in range(0,len(list1),1):
        list1[i]='./{}/'.format(path)+list1[i]
        list2[i]=list2[i][0]
    return list1,list2

def load_image(img,size):
    im = Image.open(img, 'r')
    im = im.resize(size)
    return im

def logistic_core(nparray_result):
    ans=1/(1+np.exp(nparray_result*-1))
    return ans   

def decision_core(boundary,data1,data2):
    if(abs(data1-data2)<=boundary):
        return 1

def get_pixeldata(objdata,size_x,size_y):
    im_plot_list=[]
    for i in range(0,size_y,1):
        im_plot_list_reg=[]
        for j in range(0,size_x,1):
            im_plot_list_reg.append(objdata.getpixel((j,i)))
        im_plot_list.append(im_plot_list_reg)
    return im_plot_list

boundary_mod = SourceModule("""
  __global__ void cudaboundaryfunction(float *boundarylist,float *pixel_list,int xsize,int ysize,int Boundary)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx=i+j*xsize;
    __syncthreads();
    if(i < xsize-1 && j < ysize){
        int idx2=idx+1;
        if(abs(pixel_list[idx2]-pixel_list[idx])>Boundary){
            boundarylist[idx]=1;
        }
    }
    __syncthreads();
    if(i < xsize && j < ysize-1){
        int idx3=idx+xsize;
        if(abs(pixel_list[idx3]-pixel_list[idx])>Boundary){
            boundarylist[idx]=1;
        }
    }
  }
  """)
def decision_boundary_of_obj(Boundary,pixel_list,x_col,y_row):
    func = boundary_mod.get_function("cudaboundaryfunction")
    blocksize=32
    if(x_col%blocksize != 0 and y_row%blocksize != 0):
        gridsize=(int(x_col//blocksize)+1,int(y_row//blocksize)+1,1)
    else:
        gridsize=(int(x_col//blocksize),int(y_row//blocksize),1)
    boundarylist=np.zeros_like(pixel_list).astype(np.float32)
    func(cuda.InOut(boundarylist),cuda.In(pixel_list),np.int32(x_col),np.int32(y_row),np.int32(Boundary),block=(blocksize,blocksize,1),grid=gridsize)
    return boundarylist

filter_mod = SourceModule("""
  __global__ void cudafilterfunction(float *fealist1,float *boundarylist,float *fea1,int xsize,int ysize)
  {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    if(Row < ysize && Col < xsize){
        float pixVal = 0;
        int startCol = Col - 3 / 2;
        int startRow = Row - 3 / 2;
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                int curRow = startRow + i;
				int curCol = startCol + j;
                if (curRow > -1 && curRow<ysize&&curCol>-1 && curCol < xsize)
				{
					pixVal += fea1[i*3 + j] * boundarylist[curRow*xsize + curCol];
				}
            }
        }
        fealist1[Row*xsize + Col] = pixVal;
    }
  }
  """)
def multi_filter(boundarylist,x_col,y_row,num_filters):
    boundarylist.astype(np.float32)
    func = filter_mod.get_function("cudafilterfunction")
    blocksize=3
    if(x_col%blocksize != 0 and y_row%blocksize != 0):
        gridsize=(int(x_col//blocksize)+1,int(y_row//blocksize)+1,1)
    else:
        gridsize=(int(x_col//blocksize),int(y_row//blocksize),1)
    for i in range(0,num_filters,1):
        fea1=np.loadtxt('ai_save_model/filter{}.txt'.format(i))
        fealist1=np.zeros_like(boundarylist).astype(np.float32)
        func(cuda.InOut(fealist1),cuda.In(boundarylist),cuda.In(fea1),np.int32(x_col),np.int32(y_row),block=(blocksize,blocksize,1),grid=gridsize) #size:390x60
        np.savetxt('ai_save_model/featurelist{}.txt'.format(i), fealist1.astype(np.int32),fmt='%.0f')

def loadtt(i):
    a=np.loadtxt('ai_save_model/featurelist{}.txt'.format(i))
    return a

def loadfealist(filternum):
    fea_list=[]
    for i in range(0,filternum,1):
        if(i==0):
            fea_list=loadtt(i)
        elif(i==1):
            fea_list_reg=loadtt(i)
            fea_list=np.stack((fea_list,fea_list_reg),axis=-1)
            if(filternum-1 != i):
                fea_list=fea_list.tolist()
        else:
            fea_list_reg=loadtt(i)
            for j in range(0,len(fea_list),1):
                for z in range(0,len(fea_list[j]),1):
                    fea_list[j][z].append(fea_list_reg[j][z])
            if(filternum-1 == i):
                fea_list=np.array(fea_list)
    return fea_list

def loadfilter(filternum):
    filters=[]
    for i in range(filternum):
        b=np.loadtxt('ai_save_model/filter{}.txt'.format(i)).tolist()
        filters.append(b)
    filters=np.array(filters)
    return filters

class Softmax:
    def __init__(self,input_len,nodes):
        self.weights = np.random.randn(input_len,nodes) / input_len
        self.biases = np.zeros(nodes)
    def forward(self,input):
        self.last_input_shape = input.shape #反向時用
        input = input.flatten()
        self.last_input = input #反向時用
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals #反向時用
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
    def backprop(self, d_L_d_out, learn_rate): #d_L_d_out 是該層輸出的損失梯度 目前實驗設定2個
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
        
            # Sum of all e^totals
            S = np.sum(t_exp)
            
            # Gradients of out[i] against totals
            # all gradients are given value with k != c
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2) ######待修正 ，太大導致乘法overflow(溢位)
            # change the value of k == c
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
        
            # Gradients of out[i] against totals
            # gradients to every weight in every node
            # this is not the final results
            d_t_d_w = self.last_input  # vector
            d_t_d_b = 1
            # 1000 x 10
            d_t_d_inputs = self.weights
        
            # Gradients of loss against totals
            # d_L_d_t, d_out_d_t, vector, 10 elements
            d_L_d_t = gradient * d_out_d_t
        
            # Gradients of loss against weights/biases/input
            # (1000, 1) @ (1, 10) to (1000, 10)
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1000, 10) @ (10, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
        
            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
        
            # it will be used in previous pooling layer
            # reshape into that matrix
            return d_L_d_inputs.reshape(self.last_input_shape)

def iterate_regions(image):                      #每次迭代要跑的
    h, w = image.shape
    for i in range(h - 2):
        for j in range(w - 2):
            im_region = image[i:(i + 3), j:(j + 3)]
            yield im_region, i, j #yield參考 https://chriskang028.medium.com/python-%E8%A3%A1%E7%9A%84-yield-%E8%AE%93%E4%BD%A0%E7%B0%A1%E5%96%AE-%E5%BF%AB%E9%80%9F%E7%9E%AD%E8%A7%A3-yield-%E7%9A%84%E6%A6%82%E5%BF%B5-f660521f3aa7


def conv_back(d_L_d_out, learn_rate, last_input, filters, num_filters):         # need : filters.shape,last_input,num_filters,iterate_regions,filters --> filters,last_input, iterate_regions
    
    d_L_d_filters = np.zeros(filters.shape)
        
    for im_region, i, j in iterate_regions(last_input):
        for f in range(num_filters):
            # d_L_d_filters[f]: 3x3 matrix
            # d_L_d_out[i, j, f]: num
            # im_region: 3x3 matrix in image
            d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                
    # Update filters
    filters -= learn_rate * d_L_d_filters
    for i in range(0,num_filters,1):
        np.savetxt('ai_save_model/filter{}.txt'.format(i),filters[i].astype(np.int32),fmt='%.0f')
    # We aren't returning anything here since we use Conv3x3 as
    # the first layer in our CNN. Otherwise, we'd need to return
    # the loss gradient for this layer's inputs, just like every
    # other layer in our CNN.
    return None

#init 變數
pathfor,pathname=pathfrom('ai_load_training_png')
filternum=8
initfilter(filternum)
softmax = Softmax(390 * 60 * 8, 2) # 13x13x8 -> 10
loss = 0
num_correct = 0



def forward(boundarylist,im_plot_size_x,im_plot_size_y,filternum):
    multi_filter(boundarylist,im_plot_size_x,im_plot_size_y,filternum)   #卷積(gpgpu加速)
    fea_list=loadfealist(filternum)   #載入filternum數量的增維featurelist #390x60x8
    filter_list=loadfilter(8)
    out = softmax.forward(fea_list)   #390x60x8轉187200
    loss = -np.log(out[int(im_plot_name)])
    acc = 1 if np.argmax(out) == int(im_plot_name) else 0
    # print(out.astype(np.int32),int(im_plot_name),acc) #測試數值是否正確
    return out,loss,acc,filter_list

def train(boundarylist, im_plot_size_x, im_plot_size_y, filternum, im_plot_name, lr=.005):
    last_input=boundarylist
    out, loss, acc, filter_list = forward(boundarylist,im_plot_size_x,im_plot_size_y,filternum)
    gradient = np.zeros(2)
    gradient[int(im_plot_name)] = -1 / out[int(im_plot_name)]
    gradient = softmax.backprop(gradient, lr)
    gradient = conv_back(gradient, lr, last_input, filter_list, filternum)

    return loss, acc



for pi in range(0,len(pathfor),1):
    im_plot=load_image(pathfor[pi],(390,60)).convert('L') #pixel :390x60
    im_plot_name=pathname[pi]
    im_plot_size_x,im_plot_size_y=im_plot.size
    im_plot_list=np.array(get_pixeldata(im_plot,im_plot_size_x,im_plot_size_y)).astype(np.float32)  #取得圖片pixel
    boundarylist=decision_boundary_of_obj(20,im_plot_list,im_plot_size_x,im_plot_size_y)  #邊緣柔化
    

    print(
            '[Step %d] Past 1 steps: Average Loss %.3f | Accuracy: %d%%' %
            (pi + 1, loss / 100, num_correct)
    )
    if pi % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (pi + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0


    

    l, acc = train(boundarylist, im_plot_size_x, im_plot_size_y, filternum, im_plot_name, lr=.005)

    loss += l
    num_correct += acc