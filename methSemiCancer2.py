import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pandas as pd 
import os
import numpy as np
import sys
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MyBaseDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]


class UnlabelDataset(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data
        
    def __getitem__(self, index): 
        return self.x_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]


class DomainDataset(Dataset) :
    def __init__(self, x_data, y_data, z_data):
        self.x_data = x_data
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index], self.z_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")

source_X_filename = sys.argv[1]
source_y_filename = sys.argv[2]
target_x_filename = sys.argv[3]
result_dir = sys.argv[4]

raw_x = pd.read_csv(source_X_filename, index_col = 0)
raw_y = pd.read_csv(source_y_filename)
raw_test_target_x = pd.read_csv(target_x_filename, index_col = 0)

del raw_test_target_x['Batch']
del raw_test_target_x['domain_idx']
raw_test_target_x = raw_test_target_x.values

raw_target_x = pd.read_csv(target_x_filename, index_col = 0)
raw_target_domain_y = raw_target_x['domain_idx'].tolist()

y_train = raw_y['subtype'].tolist()
num_subtype = len(set(y_train))
y_train = np.array(y_train)

del raw_target_x['domain_idx']
del raw_target_x['Batch']

raw_target_x = raw_target_x.values
x_train = raw_x.values


domain_x = np.append(x_train, raw_target_x, axis = 0)
raw_source_domain_y = np.zeros(len(y_train), dtype = int) # TCGA label : 0
domain_y = np.append(raw_source_domain_y, raw_target_domain_y)

num_domain = len(set(domain_y))

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

domain_x = torch.from_numpy(domain_x)
domain_y = torch.from_numpy(domain_y)

target_x = torch.from_numpy(raw_target_x)
target_init_y = torch.randint(low=0, high=num_subtype, size = (len(target_x),))

domain_z = torch.cat((y_train, target_init_y), 0)
raw_test_target_x = torch.from_numpy(raw_test_target_x)

num_feature = len(x_train[0])
num_train = len(x_train)
num_test = len(raw_target_x)

train_dataset = MyBaseDataset(x_train, y_train)
domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
target_dataset = MyBaseDataset(target_x, target_init_y)
test_target_dataset = UnlabelDataset(raw_test_target_x)

batch_size = 128
target_batch_size = 64
test_target_batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size, shuffle = False)
test_target_dataloader = DataLoader(test_target_dataset, batch_size = test_target_batch_size)

n_fe_embed1 = 1024
n_fe_embed2 = 512
n_cl_embed1 = 256
n_c_h1 = 256
n_c_h2 = 64
n_d_h1 = 256
n_d_h2 = 64


class FeatureExtractor(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(num_feature, n_fe_embed1),
            nn.LeakyReLU(),
            nn.Linear(n_fe_embed1, n_fe_embed2),
            nn.LeakyReLU()
            )
    def forward(self, x) :
        embedding = self.feature_layer(x)
        return embedding


class AutoEncoder(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_feature, n_fe_embed1),
            nn.LeakyReLU(),
            nn.Linear(n_fe_embed1, n_fe_embed2),
            )
        self.decoder = nn.Sequential(
            nn.Linear(n_fe_embed2, n_fe_embed1),
            nn.LeakyReLU(),
            nn.Linear(n_fe_embed1, num_feature),
            )
    def forward(self, x) :
        embedding = self.encoder(x)
        reconst_X = self.decoder(embedding)
        return embedding, reconst_X

class ContrastiveLearningModule(nn.Module) :
    def __init__(self, temperature) :
        super().__init__()
        self.temperature = temperature

    def calculate_cos_sim (self, a, b) :
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0,1))
        res_flat = res.flatten()
        return res_flat

    def forward(self, x_source, x_bc_target, x_raw_target) :
        sim_pos = self.calculate_cos_sim(x_source, x_bc_target)
        sim_neg = self.calculate_cos_sim(x_source, x_raw_target)
        cl_loss = 0.0 # NT-Xent Loss
        tmp_loss_denom = 0.0
        for i in range(len(sim_neg)) :
            tmp_loss_denom += torch.exp(sim_neg[i] / self.temperature)
        for i in range(len(sim_pos)) :
            tmp_loss_numer = torch.exp(sim_pos[i] / self.temperature)
            tmp_loss = -1 * torch.log(tmp_loss_numer/tmp_loss_denom)
            cl_loss += tmp_loss
        if cl_loss != 0.0 :
            cl_loss = cl_loss/len(sim_pos)
        return cl_loss


class DomainDiscriminator(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.disc_layer = nn.Sequential(
            nn.Linear(n_fe_embed2, n_d_h1),
            nn.LeakyReLU(),
            nn.Linear(n_d_h1, n_d_h2),
            nn.LeakyReLU(),
            nn.Linear(n_d_h2, num_domain)
            )
    def forward(self, x) :
        domain_logits = self.disc_layer(x)
        return domain_logits


class SubtypeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_fe_embed2, n_c_h1),
            nn.LeakyReLU(),
            nn.Linear(n_c_h1, n_c_h2),
            nn.LeakyReLU(),
            nn.Linear(n_c_h2, num_subtype)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


feature_extract_model = FeatureExtractor().to(device)
domain_disc_model = DomainDiscriminator().to(device)
subtype_pred_model = SubtypeClassifier().to(device)
cl_model = ContrastiveLearningModule(temperature = 0.5).to(device)
ae_model = AutoEncoder().to(device)

c_loss = nn.CrossEntropyLoss() 
domain_loss = nn.CrossEntropyLoss() 
ae_loss = nn.L1Loss()

fe_optimizer = torch.optim.Adam(feature_extract_model.parameters(), lr=1e-5)
c_optimizer = torch.optim.Adam(subtype_pred_model.parameters(), lr=1e-5)
d_optimizer = torch.optim.Adam(domain_disc_model.parameters(), lr=1e-6)
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-4)


def autoEncoder_train(epoch, dataloader, ae_model, ae_loss, ae_optimizer):
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        latent_feature, recon_X = ae_model(X)
        loss = ae_loss(X, recon_X)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0 :
        print(f"[AE Epoch {epoch+1}] loss: {total_loss:>5f}")

def pretrain_classifier(epoch, dataloader, fe_model, c_model, c_loss, fe_optimizer, c_optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        extracted_feature = fe_model(X)
        pred = c_model(extracted_feature)
        loss = c_loss(pred, y)
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        loss.backward()
        fe_optimizer.step()
        c_optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss = loss.item()
    correct /= size
    if epoch % 10 == 0 :
        print(f"[PT Epoch {epoch+1}] Training loss: {loss:>5f}, Training Accuracy: {(100*correct):>0.2f}%")



def class_alignment_train(epoch, domain_dataloader, fe_model, fe_optimizer, c_optimizer) :
    for batch, (X, y_domain, z_subtype) in enumerate(domain_dataloader):
        X, y_domain, z_subtype = X.to(device), y_domain.to(device), z_subtype.to(device)
        X = X.float()
        batch_subtype_list = z_subtype.unique()
        X_embed = fe_model(X)
        #
        align_loss = torch.zeros((1) ,dtype = torch.float64)
        align_loss = align_loss.to(device)
        #
        for subtype in batch_subtype_list :
            sample_idx_list = (z_subtype == subtype).nonzero(as_tuple = True)[0]
            if len(sample_idx_list) < 1 :
                continue
            #else :
            tmp_x = X_embed[sample_idx_list]
            tmp_y = y_domain[sample_idx_list]
            tmp_z = z_subtype[sample_idx_list]
            batch_domain_list = tmp_y.unique()
            domain_centroid_stack = []
            for domain in batch_domain_list :
                domain_idx_list = (tmp_y == domain).nonzero(as_tuple = True)[0]
                if len(domain_idx_list) != 1 :
                    tmp_x_domain = tmp_x[domain_idx_list]
                    tmp_centroid = torch.div(torch.sum(tmp_x_domain, dim = 0), len(domain_idx_list))
                    domain_centroid_stack.append(tmp_centroid)
            if len(domain_centroid_stack) == 0 :
                continue
            else :
                domain_centroid_stack = torch.stack(domain_centroid_stack)
            subtype_centroid = torch.mean(domain_centroid_stack, dim = 0)
            # Duplicate the subtype centroid to get dist with each domain_centroid
            subtype_centroid_stack = []
            for i in range(len(domain_centroid_stack)) :
                subtype_centroid_stack.append(subtype_centroid)
            subtype_centroid_stack = torch.stack(subtype_centroid_stack)
            pdist_stack = nn.L1Loss()(subtype_centroid_stack, domain_centroid_stack)
            align_loss +=  torch.mean(pdist_stack, dim = 0)
        if align_loss == 0.0 :
            continue
        align_loss = align_loss / len(batch_subtype_list)
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        align_loss.backward()
        fe_optimizer.step() 
        c_optimizer.step()
    align_loss = align_loss.item()
    if epoch % 10 == 0 :
        print(f"[CA Epoch {epoch+1}] align loss: {align_loss:>5f}\n")


def ssl_train_classifier(epoch, source_dataloader, target_dataloader, fe_model, c_model, c_loss, fe_optimizer, c_optimizer) :
    source_size = len(source_dataloader.dataset)
    target_size = len(target_dataloader.dataset)
    target_pseudo_label = torch.empty((0), dtype = torch.int64)
    target_pseudo_label = target_pseudo_label.to(device)
    #
    for batch, (target_X, target_y) in enumerate(target_dataloader):
        target_X, target_y = target_X.to(device), target_y.to(device)
        target_X = target_X.float()
        extracted_feature = fe_model(target_X)
        batch_target_pred = c_model(extracted_feature)
        batch_pseudo_label = batch_target_pred.argmax(1)
        target_pseudo_label = torch.cat((target_pseudo_label, batch_pseudo_label), 0)
        if batch == 0 :
            target_loss = c_loss(batch_target_pred, target_y)
        else :
            target_loss = target_loss + c_loss(batch_target_pred, target_y)
    target_loss = target_loss / (batch + 1)
    alpha_f = 0.01
    t1 = 100
    t2 = 200
    if epoch < t1 :
        alpha = 0
    elif epoch < t2 :
        alpha = (epoch - t1) / (t2 - t1) * alpha_f
    else :
        alpha = alpha_f
    correct = 0
    for batch, (source_X, source_y) in enumerate(source_dataloader):
        source_X, source_y = source_X.to(device), source_y.to(device)
        source_X = source_X.float()
        source_extracted_feature = fe_model(source_X)
        source_pred = c_model(source_extracted_feature)
        source_loss = c_loss(source_pred, source_y)
        ssl_loss = source_loss + alpha * target_loss
        target_loss.detach_()
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        ssl_loss.backward()
        fe_optimizer.step()
        c_optimizer.step()
        correct += (source_pred.argmax(1) == source_y).type(torch.float).sum().item()
    ssl_loss = ssl_loss.item()
    source_loss = source_loss.item()
    target_loss = target_loss.item()
    correct /= source_size
    if epoch % 10 == 0 :
        print(f"[SSL Epoch {epoch+1}] alpha : {alpha:>3f}, SSL loss: {ssl_loss:>5f}, source loss: {source_loss:>5f}, target loss: {target_loss:>4f}, source ACC: {(100*correct):>0.2f}%\n")
    return target_pseudo_label


def adversarial_train_disc(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer) :
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y, z_subtype) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        extracted_feature = fe_model(X)
        pred = d_model(extracted_feature)
        d_loss = domain_loss(pred, y)
        # Backpropagation
        fe_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    d_loss = d_loss.item()
    correct /= size 
    if t % 10 == 0 :
        print(f"[AT Epoch {epoch+1}] Disc loss: {d_loss:>5f}, Training Accuracy: {(100*correct):>0.2f}%", end = ", ")


def adversarial_train_fe(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer) :
    size = len(dataloader.dataset)
    for batch, (X, y, z_subtype) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.float()
        extracted_feature = fe_model(X)
        pred = d_model(extracted_feature)
        fake_y = torch.randint(low=0, high=num_domain, size = (len(y),))
        fake_y = fake_y.to(device)
        g_loss = domain_loss(pred, fake_y)
        # Backpropagation
        fe_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        fe_optimizer.step()
    g_loss = g_loss.item()
    if epoch % 10 == 0:
        print(f"Gen loss: {g_loss:>5f}")


def contrastive_train_fe(epoch, domain_dataloader, fe_model, cl_model, ae_model, fe_optimizer) :
    batch_count = 0
    total_loss = 0.0
    ae_model.eval()
    for batch, (X, y_domain, z_subtype) in enumerate(domain_dataloader):
        X, y_domain, z_subtype = X.to(device), y_domain.to(device), z_subtype.to(device)
        X = X.float()
        source_domain_num = 0
        source_sample_idx_list = (y_domain == source_domain_num).nonzero(as_tuple = True)[0]
        target_sample_idx_list = (y_domain != source_domain_num).nonzero(as_tuple = True)[0]
        source_x = X[source_sample_idx_list]
        target_x = X[target_sample_idx_list]
        target_raw_x_embed, _X = ae_model(target_x)
        source_x_embed = fe_model(source_x)
        target_x_embed = fe_model(target_x)
        contrast_loss = cl_model(source_x_embed, target_x_embed, target_raw_x_embed) 
        try :
            fe_optimizer.zero_grad()
            contrast_loss.backward()
            fe_optimizer.step()
            total_loss += contrast_loss.item()
            batch_count += 1
        except :
            print("None")
    total_loss /= batch_count
    if epoch % 10 == 0:
        print(f"[CL Epoch {epoch+1}] loss: {total_loss:>5f}")


def test_classifier(dataloader, fe_model, c_model, c_loss):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    fe_model.eval()
    c_model.eval()
    pred_subtype_list = []
    with torch.no_grad():
        for batch, (X) in enumerate(dataloader):
            X = X.to(device)
            X = X.float()
            extracted_feature = fe_model(X)
            pred = c_model(extracted_feature)
            pred_subtype_list.append(pred.argmax(1))
    pred_subtype_list = torch.cat(pred_subtype_list, 0)
    return pred_subtype_list


def get_embed(dataloader, fe_model, c_model) :
    fe_model.eval()
    c_model.eval()
    X_embed_list = []
    y_list = []
    with torch.no_grad() :
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            X_embed = fe_model(X)
            X_embed_list.append(X_embed)
            y_list.append(y)
    X_embed_list = torch.cat(X_embed_list, 0)
    y_list = torch.cat(y_list, 0)
    return X_embed_list, y_list


def get_embed_target(dataloader, fe_model, c_model) :
    fe_model.eval()
    c_model.eval()
    X_embed_list = []
    with torch.no_grad() :
        for batch, (X) in enumerate(dataloader):
            X = X.to(device)
            X = X.float()
            X_embed = fe_model(X)
            X_embed_list.append(X_embed)
    X_embed_list = torch.cat(X_embed_list, 0)
    return X_embed_list


def get_embed_domain(domain_dataloader, fe_model, c_model) :
    fe_model.eval()
    c_model.eval()
    X_embed_list = []
    y_list = []
    with torch.no_grad() :
        for batch, (X, y, z) in enumerate(domain_dataloader):
            X, y, z = X.to(device), y.to(device), z.to(device)
            X = X.float()
            X_embed = fe_model(X)
            X_embed_list.append(X_embed)
            y_list.append(y)
    X_embed_list = torch.cat(X_embed_list, 0)
    y_list = torch.cat(y_list, 0)
    return X_embed_list, y_list


pt_epochs = 500
ad_train_epochs = 500
ae_train_epochs = 500
cl_train_epochs = 500
ssl_train_epochs = 500
ft_epochs = 800


# 1. Pre-training
for t in range(pt_epochs):
    pretrain_classifier(t, train_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer, c_optimizer)


# 2. Adversarial training
for t in range(ad_train_epochs):
    adversarial_train_disc(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss, fe_optimizer, d_optimizer)
    adversarial_train_fe(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss, fe_optimizer, d_optimizer)


# Autoencoder training
for t in range(ae_train_epochs) :
    autoEncoder_train(t, target_dataloader, ae_model, ae_loss, ae_optimizer)

# Contrastive learning
for t in range(cl_train_epochs) :
    contrastive_train_fe(t, domain_dataloader, feature_extract_model, cl_model, ae_model, fe_optimizer)

# 3. SSL training
for t in range(ssl_train_epochs) :
    target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
    target_dataset = MyBaseDataset(target_x, target_pseudo_label)
    target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size)


# 4. Fine-tuning (AD + SSL + CA)
for t in range(ft_epochs) :
    # SSL
    target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
    target_dataset = MyBaseDataset(target_x, target_pseudo_label)
    target_dataloader = DataLoader(target_dataset, batch_size = target_batch_size)

    # CA
    target_pseudo_label = target_pseudo_label.to("cpu")
    domain_z = torch.cat((y_train, target_pseudo_label), 0)
    domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
    domain_dataloader = DataLoader(domain_dataset, batch_size = batch_size, shuffle = True)
    class_alignment_train(t, domain_dataloader, feature_extract_model, fe_optimizer, c_optimizer)
    
test_target_pred_ft  = test_classifier(test_target_dataloader, feature_extract_model, subtype_pred_model, c_loss)
test_target_pred_ft = test_target_pred_ft.detach().cpu().numpy()

pd.DataFrame({'pred_subtype' : test_target_pred_ft}).to_csv(os.path.join(result_dir, "target_prediction.csv"), mode = "w", index = False)
#np.savetxt(os.path.join(result_dir, "target_prediction.csv"), test_target_pred_ft, fmt="%.0f", delimiter=",")

'''
source_X_embed, res_y = get_embed(train_dataloader, feature_extract_model, subtype_pred_model)
source_X_embed = source_X_embed.detach().cpu().numpy()
pd.DataFrame(source_X_embed).to_csv(os.path.join(result_dir , "ft_source_X_embed_group_" + group_num + ".csv"), mode = "w", index = False)
res_y = res_y.detach().cpu().numpy()
np.savetxt(os.path.join(result_dir, "ft_source_y_group_" + group_num + ".csv"), res_y, fmt="%.0f", delimiter=",")


target_X_embed = get_embed_target(test_target_dataloader, feature_extract_model, subtype_pred_model)
target_X_embed = target_X_embed.detach().cpu().numpy()
pd.DataFrame(target_X_embed).to_csv(os.path.join(result_dir , "ft_target_X_embed_group_" + group_num + ".csv"), mode = "w", index = False)


target_X_embed, res_y = get_embed_domain(domain_dataloader, feature_extract_model, subtype_pred_model)
target_X_embed = target_X_embed.detach().cpu().numpy()
pd.DataFrame(target_X_embed).to_csv(os.path.join(result_dir , "ft_domain_X_embed_group_" + group_num + ".csv"), mode = "w", index = False)
res_y = res_y.detach().cpu().numpy()
np.savetxt(os.path.join(result_dir, "ft_domain_y_group_" + group_num + ".csv"), res_y, fmt="%.0f", delimiter=",")
'''
