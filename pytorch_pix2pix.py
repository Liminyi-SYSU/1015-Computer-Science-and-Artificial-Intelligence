import os, time, pickle, argparse, network, util
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

#创建需要的参数，使用argparse库#

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades',  help='')
parser.add_argument('--train_subfolder', required=False, default='train',  help='')
parser.add_argument('--test_subfolder', required=False, default='val',  help='')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=5, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')#该参数必须在命令行中出现#
parser.add_argument('--train_epoch', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')#该参数必须在命令行中出现#
opt = parser.parse_args()
print(opt)

#参数创建完毕#

#设置结果的存储路径#
root = opt.dataset + '_' + opt.save_root + '/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

#数据上传#
transform = transforms.Compose([
        transforms.ToTensor()#将传输进的取值为[0,255]图像转换为torch.FloadTensor,形状为[Channel,Height,Width],取值范围[0,1.0]#
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))#给定RGB的均值和方差，将tensor正则化#
]) #图片像素值归一化操作，不改变图像本身携带的信息，方便后期处理

train_loader = util.data_load('data/' + opt.dataset, opt.train_subfolder, transform, opt.batch_size, shuffle=True)
test_loader = util.data_load('data/' + opt.dataset, opt.test_subfolder, transform, opt.test_batch_size, shuffle=True)#装载图片数据#
test = test_loader.__iter__().__next__()[0]
img_size = test.size()[2]
if opt.inverse_order:
    fixed_y_ = test[:, :, :, 0:img_size]  #装载的数据是原图和模糊图的连接图像，2:1的图形，在这里切割#
    fixed_x_ = test[:, :, :, img_size:]
else:
    fixed_x_ = test[:, :, :, 0:img_size]
    fixed_y_ = test[:, :, :, img_size:]

if img_size != opt.input_size:
    fixed_x_ = util.imgs_resize(fixed_x_, opt.input_size)
    fixed_y_ = util.imgs_resize(fixed_y_, opt.input_size)#图片尺寸不符时将图片尺寸重塑成参数标定的尺寸#

#建立GAN网络部分
G = network.generator(opt.ngf)#设置生成器，参数为ngf#
D = network.discriminator(opt.ndf)#设置识别器，参数为ndf#
G.weight_init(mean=0.0, std=0.02)#权重初始化，均值0，方差0.02#
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G.train()
D.train()

#损失函数
BCE_loss = nn.BCELoss().cuda()#BCE-loss传递给GPU#
L1_loss = nn.L1Loss().cuda()#L1-loss传递给GPU#

#调用pytorch.optim实现Adam算法#
G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))#包含生成器所有参数的迭代器，学习率，计算梯度以及梯度平方的运行平均值的系数#
D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))#包含识别器所有参数的迭代器，学习率，计算梯度以及梯度平方的运行平均值的系数#

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(opt.train_epoch):#开始循环，至epoch到达设定的参数时截止#
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()#识别器内模型参数梯度设置为0#

        if opt.inverse_order:
            y_ = x_[:, :, :, 0:img_size]#装载的数据是原图和模糊图的连接图像，2:1的图形，在这里切割#
            x_ = x_[:, :, :, img_size:]
        else:
            y_ = x_[:, :, :, img_size:]
            x_ = x_[:, :, :, 0:img_size]
            
        if img_size != opt.input_size:
            x_ = util.imgs_resize(x_, opt.input_size)
            y_ = util.imgs_resize(y_, opt.input_size)#图片尺寸不符时将图片尺寸重塑成参数标定的尺寸#

        if opt.resize_scale:
            x_ = util.imgs_resize(x_, opt.resize_scale)
            y_ = util.imgs_resize(y_, opt.resize_scale)

        if opt.crop_size:
            x_, y_ = util.random_crop(x_, y_, opt.crop_size)#任意选取切割图片的中心点#

        if opt.fliplr:
            x_, y_ = util.random_fliplr(x_, y_)#将x，y左右翻转#

        x_, y_ = Variable(x_.cuda()), Variable(y_.cuda()) #将先前处理过的x_,y_搬运至GPU并包含进Variable类方便求梯度（后面要算梯度）

        D_result = D(x_, y_).squeeze() #消除shape=1的量后丢给D_result#
        D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) #计算真实损失，将上一行的D_result.size化为全为1的张量

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).cuda())) #计算虚假损失

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5 #得到训练后的最终损失
        D_train_loss.backward() #计算识别器训练损失函数的梯度和
        D_optimizer.step() #单次优化，重复运算

        train_hist['D_losses'].append(D_train_loss.data[0])        #此处有一个版本问题，data[0]改为item（）

        D_losses.append(D_train_loss.data[0])

        #训练生成器
        G.zero_grad()#生成器内模型参数梯度设置为0#

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()

        G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda())) + opt.L1_lambda * L1_loss(G_result, y_)#计算生成器中的损失#
        G_train_loss.backward()#计算生成器训练损失函数的梯度和
        G_optimizer.step()#单次优化，重复运算

        train_hist['G_losses'].append(G_train_loss.data[0])

        G_losses.append(G_train_loss.data[0])

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time#计算训练时间#

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    util.show_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, (epoch+1), save=True, path=fixed_p)
#存图版本：
    util.show_result(G, Variable(fixed_x_.cuda(), volatile=True), fixed_y_, (epoch+1), save=True, path_1=fixed_p, path_2=fixed_q)
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), root + model + 'generator_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

util.show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
util.generate_animation(root, model, opt)
