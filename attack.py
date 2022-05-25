import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as tf
import cv2
import net_sphere
from torch.autograd import Variable
import os
import shutil
from matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img, src_pts):
    # src_img:输入的图像
    # src_pts:矩阵，经函数处理后为仿射变换的变换矩阵
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)
    # 将src_pts 变为五行二列的矩阵
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    # 将 src_pts 和 ref_pts 的数据类型转换为 float32
    tfm = get_similarity_transform_for_cv2(s, r)
    # tfm: 通过此函数寻找能够直接使用的仿射变换矩阵,详见具体函数
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    # face_img:通过opencv的函数将原图像进行一定的变换并返回
    # cv2.imshow('test', src_img)
    return face_img


def criterion(net, x, x_adv):
    # x, x_adv : torch.Tensor with shape [1,3,112,96](即代表以该维度的矩阵为原型的张量)
    # net:是由net_sphere中导入的一个标签为"sphere20a"的内容，其实是个类，详细内容见有关代码
    best_thresh = 0.3
    # best_thresh:显而易见，这是个罚函数的阈值
    img_list = [x, x_adv]
    # img:将 img_list 竖向堆叠得到的新张量
    # torch.no_grad():即不需要被求导
    img = torch.vstack(img_list)
    with torch.no_grad():
        # 将 img 转换为float模式的张量并返回(cuda加速)
        img = Variable(img.float()).cuda()
    output = net(img)
    f1, f2 = output.data
    cos_distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    # cos_distance:计算 f1,f2 俩向量夹角的余弦值(存疑)，是罚函数的判断依据
    D = torch.norm(x - x_adv)
    # D:计算x与x_adv二者的距离，本质上是两张图的差别
    C = 0 if cos_distance < best_thresh else float('inf')
    # C在经过判断后被赋值为0或者无穷
    return D + C


def generate_adversarial_face(net, x, num):
    _, _, H, W = x.shape
    m = 3 * 45 ** 2
    k = m // 20
    C = torch.eye(m)
    # C:生成m*m的对角值为1，其余为0的矩阵并返回
    c_c = 0.01
    p_c = torch.zeros(m)
    # p_c:生成m*m的零矩阵
    c_cov = 0.001
    sigma = 0.01
    success_rate = 0
    mu = 1
    x_adv = torch.randn_like(x)
    process = []
    process_i=[]
    process.append(195.4611)
    process_i.append(0)
    # x_adv:大小为m的张量，其由均值为0，方差为1的标准正态分布填充
    for i in range(0, num):
        z = MultivariateNormal(loc=torch.zeros([m]), covariance_matrix=(sigma ** 2) * C).rsample()
        # MultivariateNormal:从多元正态分布中随机抽取样本
        zero_idx = np.argsort(-C.diagonal())[k:]
        # np.argsort:返回的是元素值从小到大排序后的索引值的数组
        # diagonal: 返回指定的对角线
        z[zero_idx] = 0
        z = z.reshape([1, 3, 45, 45])
        img_size = (H, W)
        z_ = tf.interpolate(z, img_size, mode='bilinear')
        # interpolate:通过 mode 选择上采样算法进行上采样(即让图像变成更高分辨率)
        z_ = z_ + mu * (x - x_adv)
        L_after = criterion(net, x, x_adv + z_)
        L_before = criterion(net, x, x_adv)
        # 显然，这两个函数是分别得到在修改分辨率前后分别处理图片的差异度
        if L_after < L_before:
            x_adv += z_
            # 如果修改后效果更好，那就执行操作
            p_c = (1 - c_c) * p_c + np.sqrt(2 * (2 - c_c)) * z.reshape(-1) / sigma
            C[range(m), range(m)] = (1 - c_cov) * C.diagonal() + c_cov * p_c * p_c
            # 这两行计算见论文3.2.3，用于更新协方差矩阵，目测通过改变协方差矩阵影响上面的z，即随机抽取样本
            success_rate += 1
            process.append(torch.norm(x - x_adv).item())
            process_i.append(i)

        if i % 10 == 0:
            mu = mu * np.exp(success_rate / 10 - 1 / 5)
            success_rate = 0
        # 十个一组分割，根据已有成功率改变参数，目测影响z_，即高分辨率图片
        if i % 500 == 0:
            t_norm=torch.norm(x - x_adv)
            save_img(x_adv, 'iter_' + str(i) + str(t_norm))
            print(t_norm)
        print(i)
    process_i.append(4500)
    process.append(torch.norm(x - x_adv).item())
    plt.title('attack progress')
    plt.xlabel('query number')
    plt.ylabel('Frobenius norm')
    plt.xlim((0, 4500))
    plt.ylim((0, 195))
    plt.plot(process_i,process)
    plt.show()
    return x_adv


def save_img(x, name):
    x = (x * 128 + 127.5).type(torch.int)
    x = x[0].permute(1, 2, 0)
    # permute:改变tensor的维度，在这里是把前两个维度调换
    cv2.imwrite("./pic/" + name + '.png', np.array(x))
    # opencv自带的写图片


# 函数功能:选择图片
def subjective_select_image(landmark, pairs_lines, data_path, img_num):
    idx = img_num
    pair = pairs_lines[idx]
    line = pair.replace('\n', '')
    line = line.split('\t')
    name = line[0] + '/' + line[0] + '_' + '{:04}.jpg'.format(int(line[1]))
    img = cv2.imread(data_path + '/lfw/' + name)
    img = alignment(img, landmark[name])
    img = img.transpose(2, 0, 1).reshape((1, 3, 112, 96))
    img = (img - 127.5) / 128.0
    img = torch.tensor(img)
    return img


def main():
    shutil.rmtree('pic')
    os.mkdir('pic')
    net = getattr(net_sphere, 'sphere20a')()
    net.load_state_dict(torch.load('./model/sphere20a_20171020.pth'))
    net.cuda()
    net.eval()
    net.feature = True
    data_path = 'LFW'
    landmark = {}
    with open(data_path + '/lfw_landmark.txt') as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        line_num = line.replace('\n', '').split('\t')
        landmark[line_num[0]] = [int(k) for k in line_num[1:]]
    with open(data_path + '/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]
    print('请输入图片数字')
    img_num = int(input())
    x = subjective_select_image(landmark, pairs_lines, data_path, img_num)
    save_img(x, 'original')
    x_adv = generate_adversarial_face(net, x, num=4500)
    save_img(x_adv, 'final')
main()
