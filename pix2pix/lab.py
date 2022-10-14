class my_class(object):
    pass
import os
from PIL import Image
from torchvision import transforms
from torchvision import datasets
import torch
# data_loader-----------------------------------------------------------------------------------------------------------------
dataset = 'facades'
train_subfolder = 'train'
test_subfolder = 'test'
batch_size = 4
test_batch_size = 4
inverse_order = True
input_size = 256
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
img_list = os.listdir(os.path.join('data', dataset, train_subfolder))
del img_list[0]
img_dir = 'data/' + dataset + '/' + train_subfolder + '/'
#print(img_list)
'''
for i in range(len(img_list)):
    im = Image.open(img_dir + img_list[i])
    im = im.resize((256, 256))
    im.save(img_dir + img_list[i])
'''
def data_load(path, subfolder, transform, batch_size, shuffle=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]
    
    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

train_loader = data_load('data/' + dataset, train_subfolder, transform, batch_size, shuffle=True)
#test_loader = data_load('data/' + dataset, test_subfolder, transform, test_batch_size, shuffle=True)
#print(train_loader)
train = train_loader.__iter__().__next__()[0]
#print(train)
img_size = train.size()[2]
print(img_size)


if inverse_order:
    fixed_y_ = train[:, :, :, 0:img_size]
    fixed_x_ = train[:, :, :, img_size:]
else:
    fixed_x_ = train[:, :, :, 0:img_size]
    fixed_y_ = train[:, :, :, img_size:]
'''if img_size != input_size:
    fixed_x_ = util.imgs_resize(fixed_x_, opt.input_size)
    fixed_y_ = util.imgs_resize(fixed_y_, opt.input_size)'''

print(fixed_y_)
'''
path = 'data/' + dataset
dset = datasets.ImageFolder(path, transform)
print(dset)
ind = dset.class_to_idx[train_subfolder]
#print(ind)
type(ind)
'''
