#This neural network (beta-variational autoencoder) generates a video prediction on the foundation of a sequence of provided images.
#Hereby, each image in a given sequence incorporates nine historical logarithmic returns (3x3 image, one pixel = one log return),
#associated with nine S&P500 stocks, captured at one point in time. After the video prediction has been generated, the pixel values of the produced frames
#are transformed back into logarithmic returns which in turn are converted back into adjusted closing prices to provide (nine, one forecast for each financial asset)
#ten-step ahead forecasts. Please note that this neural network has been inspired by the work of Franchesci et al. who have developed a state-space variational autoencoder model,
#named Stochastic Latent Residual Video Prediction, which was presented in their paper:

#Franchesci, J.-Y., Delasalles, E., Chen, M., Lamprier, S., and Gallinari, P. (2020). Stochastic Latent Residual Video Prediction.
#https://arxiv.org/abs/2002.09219.

#The Python code for their model has been provided in the GitHub repository of Edouard Delasalles:

#Delasalles, E. (2020). Official implementation of the paper Stochastic Latent Residual Video Prediction.
#https://github.com/edouardelasalles/srvp.

#The idea of treating a financial time series forecasting task as a video prediction problem as well as a possible price movement direction metric for performance evaluation
#has been described in the work of Zeng et al.:

#Zeng, Z., Balch, T., and Veloso, M. (2021). Deep Video Prediction for Time Series Forecasting.
#https://arxiv.org/abs/2102.12061.

#Import all necessary libraries

import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
from functools import partial
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_percentage_error
# %matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Download the dataset
#Use the code below (commented out) if the download should happen directly through Yahoo Finance

'''
ticker_list = ['AAPL', 'ACN', 'ADBE',
              'AMD', 'GOOG', 'MU',
              'PYPL', 'QCOM', 'STX']
stock_prices = yf.download(ticker_list, start = '2021-01-15', end = '2023-12-16', interval = '1d')['Adj Close']
'''

stock_prices = pd.read_csv('https://github.com/sch-wm/time_series/raw/main/yfinance_stock_prices_for_neural_net.csv', index_col = 0)
stock_prices.index = pd.to_datetime(stock_prices.index)

#Inspect the data

stock_prices.head(10)
stock_prices.shape

#Preprocess the data (insert missing dates and remove the related nans through linear interpolation)

dates = pd.date_range(start = pd.to_datetime('2021-01-15').tz_localize("GMT+0") , end = pd.to_datetime('2023-12-16').tz_localize("GMT+0"), freq = 'B')
stock_prices = stock_prices.reindex(dates)
stock_prices.interpolate(inplace = True)

#Inspect the data

stock_prices.shape

#Use ln on the prices and then difference once to get log returns manually

log_stock_prices = np.log(stock_prices) #761 data points, index: 0 - 760, data starts at Friday, ends at Friday
log_returns = log_stock_prices.diff() #One data point gets lost after differencing
log_returns.dropna(inplace = True) #760 data points, 760 divisible by 5 (each image batch will include 5 frames), index: 0 - 759, data starts at Monday, ends at Friday

#Define the sigmoid function

def sigmoid_to_greyscale(x):
  return (1 / (1 + np.exp(-x)))

#Define the inverse sigmoid function

def inverse_sigmoid(input):
  x = np.log(input / (1 - input))
  return x

#Transform the log returns into values from (0, 1), prepare tensors for processing

data_srvp = log_returns.apply(sigmoid_to_greyscale)
data_srvp = torch.tensor(data_srvp.values)
data_srvp = data_srvp.float()
data_srvp_5D = data_srvp.reshape(1, 760, 1, 3, 3)
data_srvp_5D = data_srvp_5D.permute(1, 0, 2, 3, 4)
train_data = data_srvp_5D[:750]
cond_data = data_srvp_5D[745:750]

###Define all utility functions

#Initialize weights of encoder, decoder and residual network, other networks get initialized by torch default initialization

def init_weights(module, init_type, init_gain):
  classname = module.__class__.__name__
  if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, init_gain)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(module.weight.data, init_gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
  elif classname == 'BatchNorm2d':
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, init_gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

#Make a normal distribution with given parameters mu and sigma

def make_n_dist(par):
  mu, raw_sigma = torch.chunk(par, 2, -1) #Split parameters into two (mu/sigma)
  assert mu.shape[-1] == raw_sigma.shape[-1]
  sigma = F.softplus(raw_sigma) + 1e-8 #Make sure sigma is positive and not zero
  n_dist = distrib.Normal(mu, sigma)
  return n_dist

#Draw a reparametrized sample from a a normal distribution

def rsample_from_n_dist(par):
  n_dist = make_n_dist(par) #Use make_n_dist function from above
  sample = n_dist.rsample()
  return sample

#Compute negative log likelihood

def log_likelihood(mu, data):
  obs_dist = distrib.Normal(mu, 1)
  log_likelihood = - obs_dist.log_prob(data) #(negative) log likelihood
  return log_likelihood

###Define network components via classes

#General encoder class for srvp

class encoder_h(nn.Module):
#nc = number of channels
#nf = number of filters
#dim_x_tilde = dimension of encoder output (dimension of latent representations)
  def __init__(self, nc, nf, dim_x_tilde):
    super(encoder_h, self).__init__()
    self.nc = nc
    self.nf = nf
    self.dim_x_tilde = dim_x_tilde
    self.conv = nn.ModuleList([nn.Conv2d(nc, nf, kernel_size = 3, stride = 1, padding = 1, bias = False),
                               nn.BatchNorm2d(nf),
                               nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                               nn.Conv2d(nf, nf, kernel_size = 3, stride = 1, padding = 1, bias = False),
                               nn.BatchNorm2d(nf),
                               nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                               ##########################################
                               nn.Conv2d(nf, nf * 2, kernel_size = 3, stride = 1, padding = 1, bias = False),
                               nn.BatchNorm2d(nf * 2),
                               nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                               nn.Conv2d(nf * 2, nf * 2, kernel_size = 3, stride = 1, padding = 1, bias = False),
                               nn.BatchNorm2d(nf * 2),
                               nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                               ##########################################
                               nn.Conv2d(nf * 2, dim_x_tilde, kernel_size = 3, stride = 1, padding = 0, bias = False),
                               nn.BatchNorm2d(dim_x_tilde),
                               nn.Tanh()])
#x is encoder input with 4D shape: sequence length (timesteps) * batch size (number of videos), channels, height, width
  def forward(self, x):
    x_tilde = x
    for layer in self.conv:
      x_tilde = layer(x_tilde)
  #Flatten encoder output to 1D
    x_tilde = x_tilde.view(-1, self.dim_x_tilde)
    return x_tilde

#General decoder class for srvp

class decoder_g(nn.Module):
#dim_y = dimension of latent states y
#nf = number of filters
#nc = number of channels
  def __init__(self, dim_y, nf, nc):
    super(decoder_g, self).__init__()
    self.dim_y = dim_y
    self.nf = nf
    self.nc = nc
    self.upconv = nn.ModuleList([nn.ConvTranspose2d(dim_y, nf * 2, kernel_size = 3, stride = 1, padding = 0, bias = False),
                                 nn.BatchNorm2d(nf * 2),
                                 nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                 ##########################################
                                 nn.ConvTranspose2d(nf * 2, nf * 2, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                 nn.BatchNorm2d(nf * 2),
                                 nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                 nn.ConvTranspose2d(nf * 2, nf, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                 nn.BatchNorm2d(nf),
                                 nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                 ##########################################
                                 nn.ConvTranspose2d(nf, nf, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                 nn.BatchNorm2d(nf),
                                 nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                 nn.ConvTranspose2d(nf, nc, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                 nn.BatchNorm2d(nc)])
  def forward(self, y):
  #x_hat depicts reconstructed images
    x_hat = y.view(* y.shape, 1, 1)
    for layer in self.upconv:
      x_hat = layer(x_hat)
    #Push 5D data through sigmoid
    x_hat = torch.sigmoid(x_hat)
    return x_hat

#General mlp classes for srvp (inference + compute residuals)

class mlp_inf(nn.Module):
#n_inp = mlp input neurons
#n_hid = mlp hidden neurons
#n_out = mlp output neurons
  def __init__(self, n_inp, n_hid, n_out):
    super(mlp_inf, self).__init__()
    self.n_inp = n_inp
    self.n_hid = n_hid
    self.n_out = n_out
    self.mlp_modules = nn.ModuleList([nn.Linear(n_inp, n_hid),
                                      nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                      nn.Linear(n_hid, n_hid),
                                      nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                      nn.Linear(n_hid, n_out)])
  def forward(self, mlp_input):
    data = mlp_input
    for layer in self.mlp_modules:
      data = layer(data)
    return data

class mlp_res(nn.Module):
#n_inp = mlp input neurons
#n_hid = mlp hidden neurons
#n_out = mlp output neurons
  def __init__(self, n_inp, n_hid, n_out):
    super(mlp_res, self).__init__()
    self.n_inp = n_inp
    self.n_hid = n_hid
    self.n_out = n_out
    self.mlp_modules = nn.ModuleList([nn.Linear(n_inp, n_hid),
                                      nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                      nn.Linear(n_hid, n_hid),
                                      nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                      nn.Linear(n_hid, n_hid),
                                      nn.LeakyReLU(negative_slope = 0.02, inplace = True),
                                      nn.Linear(n_hid, n_out)])
  def forward(self, mlp_input):
    data = mlp_input
    for layer in self.mlp_modules:
      data = layer(data)
    return data

#srvp class, inspired by the stochastic residual video prediction model, created by Franchesci, Delasalles, Chen, Lamprier and Gallinari

class SRVP(nn.Module):
#nx = size of images
#nc = number of channels
#nf = number of filters
#dim_x_tilde = dimension of encoder output
#dim_y = dimension of latent states y
#dim_z = dimension of latent state dynamics z
#nt_y1 = number of timesteps for initial latent state, inference
#n_hid_inf = number of hidden neurons in inference networks
#n_hid_res = number of hidden neurons in residual networks
  def __init__(self, nx, nc, nf, dim_x_tilde,
               dim_y, dim_z, nt_y1,
               n_hid_inf, n_hid_res):
    super().__init__()

  ###Class attributes

    self.nx = nx
    self.nc = nc
    self.nf = nf
    self.dim_x_tilde = dim_x_tilde
    self.dim_y = dim_y
    self.dim_z = dim_z
    self.nt_y1 = nt_y1
    self.n_hid_inf = n_hid_inf
    self.n_hid_res = n_hid_res

  ###Class components

    self.encoder_h_phi = encoder_h(self.nc, self.nf, self.dim_x_tilde)
    self.decoder_g_theta = decoder_g(self.dim_y, self.nf, self.nc)
    self.inf_y1_par = mlp_inf(self.dim_x_tilde * self.nt_y1, self.n_hid_inf, dim_y * 2)
    self.inf_temporal = nn.LSTM(self.dim_x_tilde, n_hid_inf, 1)
    self.inf_qz_par = nn.Linear(n_hid_inf, self.dim_z * 2)
    self.get_pz_par = mlp_inf(self.dim_y, n_hid_res, self.dim_z * 2)
    self.compute_residual = mlp_res(self.dim_y + self.dim_z, self.n_hid_res, self.dim_y)

  ###Class methods

  #Initialize weights of encoder, decoder and residual network, other networks get initialized by torch default initialization
  def init_weights_model(self):
    init_enc_dec_weights = partial(init_weights, init_type = 'normal', init_gain = 0.02)
    self.encoder_h_phi.apply(init_enc_dec_weights)
    self.decoder_g_theta.apply(init_enc_dec_weights)
    init_res_weights = partial(init_weights, init_type = 'orthogonal', init_gain = 1.41)
    self.compute_residual.apply(init_res_weights)

  #Encode the images with encoder network h
  #Compress images into latent representations

  def encode(self, x):
  #x = frames
  #nt = number of timesteps
  #nv = number of video batches
  #x_shape = shape of frames
    nt = x.shape[0]
    nv = x.shape[1]
    x_shape = x.shape[2:]
  #Flatten temporal dimension = make 4D out of 5D by making new dim -> nt * nv
    x_flat = x.view(nt * nv, * x_shape) # 4 dims for CNN-encoder
    x_tilde_flat = self.encoder_h_phi(x_flat)
  #Reverse flattening = make 3D out of 2D
    x_tilde = x_tilde_flat.view(nt, nv, self.dim_x_tilde) # reverse temporal flattening here
    return x_tilde

  #Decode latent states to receive frames

  def decode(self, y): #y is 3D (timesteps, batch size, dim_y)
    nt = y.shape[0]
    nv = y.shape[1]
    #Flatten temporal dimension
    y_flat = y.view(nt * nv, self.dim_y)
    dec_inp = y_flat
    #When 3D data is provided, decoder transforms data into 4D data in its forward function
    #4D -> (number of timesteps * number of batches, dim_y, 1, 1)
    x_hat_flat = self.decoder_g_theta(dec_inp)
    #Reverse flattening = split number of timesteps * number of batches into number of timesteps, number of batches
    x_hat = x_hat_flat.view(nt, nv, * x_hat_flat.shape[1:])
    return x_hat

  #Infer initial latent state

  def inf_y1(self, x_tilde_at_t):
    #Permute encoding to get one initial latent state per batch: timesteps_y1, video batches, dim_enc_out
    #-> video batches, timesteps_y1 * dim_enc_out
    y1_par = self.inf_y1_par(x_tilde_at_t.permute(1, 0, 2).reshape(x_tilde_at_t.shape[1], self.nt_y1 * self.dim_x_tilde))
    y1 = rsample_from_n_dist(y1_par)
    return y1, y1_par

  #Infer latent state dynamics with input from LSTM

  def inf_z_with_q(self, lstm_out):
    qz_par = self.inf_qz_par(lstm_out)
    z = rsample_from_n_dist(qz_par)
    return z, qz_par
  def residual_step(self, y_at_t, z_at_t_plus_1):
    inp_residual = torch.cat([y_at_t, z_at_t_plus_1], 1)
    #print(inp_residual.shape, 'Input_residual_shape:')
    residual_at_t_plus_1 = self.compute_residual(inp_residual)
    #print(residual_at_t_plus_1.shape, 'Output_residual_shape:')
    y_at_t_plus_1 = y_at_t + residual_at_t_plus_1
    return y_at_t_plus_1, residual_at_t_plus_1

  #Generate a sequence of latent states

  def generate(self, y1, x_tilde, nt):
    list_y = [y1]
    list_z = []
    list_qz_par = []
    list_pz_par = []
    list_residuals = []
    y_at_t = y1
    #Use LSTM to infer temporal relationships on all encoded frames in one go
    if len(x_tilde) > 0:
      temporal_for_z = self.inf_temporal(x_tilde)[0]
    else:
      temporal_for_z = []
    for time in np.linspace(start = 1, stop = (nt - 1), num = (nt - 1)):
      #print(y_at_t.shape, 'Input_pz_shape:')
      pz_par_at_t = self.get_pz_par(y_at_t)
      #print(pz_par_at_t.shape, 'Output_pz_shape:')
      list_pz_par.append(pz_par_at_t)
      if time < len(x_tilde):
        #print(temporal_for_z[int(time)].shape, 'Input_qz_shape:')
        z_at_t_plus_1, qz_par_at_t = self.inf_z_with_q(temporal_for_z[int(time)])
        #print(qz_par_at_t.shape, 'Output_qz_shape:')
        list_qz_par.append(qz_par_at_t)
      else:
        assert not self.training
        z_at_t_plus_1 = rsample_from_n_dist(pz_par_at_t)
      list_z.append(z_at_t_plus_1)
      y_at_t_plus_1, residual_at_t_plus_1 = self.residual_step(y_at_t, z_at_t_plus_1)
      y_time_t = y_at_t_plus_1
      list_y.append(y_time_t)
      list_residuals.append(residual_at_t_plus_1)
    list_y = torch.stack(list_y)
    list_z = torch.stack(list_z) if len(list_z) > 0 else None
    list_qz_par = torch.stack(list_qz_par) if len(list_qz_par) > 0 else None
    list_pz_par = torch.stack(list_pz_par) if len(list_pz_par) > 0 else None
    list_residuals = torch.stack(list_residuals)
    return list_y, list_z, list_qz_par, list_pz_par, list_residuals

  #Forward function of the video prediction neural network

  def forward(self, x, nt):
    x_tilde = self.encode(x)
    #print(x_tilde[:self.nt_y1].shape, 'Input_y1_shape:')
    y1, qy1_par = self.inf_y1(x_tilde[:self.nt_y1])
    #print(qy1_par.shape, 'Output_y1_shape:')
    list_y, list_z, list_qz_par, list_pz_par, list_residuals = self.generate(y1, x_tilde, nt)
    x_hat = self.decode(list_y)
    return x_hat, list_y, list_z, qy1_par, list_qz_par, list_pz_par, list_residuals

#Prepare dataloaders

class train_cond_datasets(Dataset):
  def __init__(self, training_cond_data):
    self.training_cond_data = training_cond_data
  def __len__(self):
    return len(self.training_cond_data)
  def __getitem__(self, idx):
    return self.training_cond_data[idx]

train_dataset = train_cond_datasets(train_data)
train_dataloader = DataLoader(train_dataset, batch_size = 5, shuffle = False)

cond_dataset = train_cond_datasets(cond_data)
cond_dataloader = DataLoader(cond_dataset, batch_size = 5, shuffle = False)

#Prepare training loop

def train_model(forward_function, batch, device, optimizer):
  optimizer.zero_grad()
  x = batch.to(device)
  nt, nv = x.shape[0], x.shape[1]

  #Inference

  x_hat, list_y, list_z, qy1_par, list_qz_par, list_pz_par, list_residuals = forward_function(x, nt)
  log_like = log_likelihood(x_hat, x).sum()
  qy1_dist = make_n_dist(qy1_par)
  kl_qy1 = distrib.kl_divergence(qy1_dist, distrib.Normal(0, 1)).sum()
  q_z_dist = make_n_dist(list_qz_par)
  p_z_dist = make_n_dist(list_pz_par)
  kl_z = distrib.kl_divergence(q_z_dist, p_z_dist).sum()

  #Compute ELBO loss

  evidence_lower_bound = log_like + 0.9 * kl_qy1 + 0.99 * kl_z
  evidence_lower_bound = evidence_lower_bound / nv
  evidence_lower_bound.backward()
  optimizer.step()
  with torch.no_grad():
    evidence_lower_bound = evidence_lower_bound.item()
    print(evidence_lower_bound, 'ELBO loss:')
    log_like = log_like.sum().item() / nv
    print(log_like, 'NLL:')
    kl_qy1 = kl_qy1.item() / nv
    print(kl_qy1, 'KLqy1:')
    kl_z = kl_z.item() / nv
    print(kl_z, 'KLz:')
  return evidence_lower_bound, log_like, kl_qy1, kl_z

#Start training and video prediction/forecasting via a main function

def main():
  device = torch.device('cpu')
  epochs = 35
  list_of_random_seeds = []
  #nseeds = len(list_of_random_seeds)
  nseeds = 1
  for random_seed in range (nseeds):
    seed = np.random.randint(1000)
    list_of_random_seeds.append(seed)
  vid_pred_seeds_append = [0.00] * 90
  for random_seed in list_of_random_seeds:
    torch.manual_seed(random_seed)
    model = SRVP(3, 1, 1, 4,
                  1, 1, 5,
                  2, 4)
    model.init_weights_model()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    forward_function = model
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003)

    #Start training
    for epoch in range(epochs):
      for batch in train_dataloader:
        model.train()
        evidence_lower_bound, log_like, kl_qy1, kl_z = train_model(forward_function, batch, device, optimizer) #Start training loop

    #Start video prediction
    ny_pred = 11
    nt_cond = 5
    nfore = 100
    for batch in cond_dataloader:
      model.eval()
      x_cond = batch.to(device)
      vid_pred_samples_append = [0.00] * 90
      for sample in range(nfore):
        x_cond_hat, y_cond, _, _, _, _, _ = model(x_cond, nt_cond)
        y_pred1 = y_cond[-1]
        y_pred = model.generate(y_pred1, [], ny_pred)[0]
        y_pred = y_pred[1:].contiguous() #Drop first latent state, as it is only used to set up the video prediction (10 latent states remain = 10 images = 10 timesteps)
        vid_pred = model.decode(y_pred).clamp(0, 1)
        vid_pred = inverse_sigmoid(vid_pred.detach()) #Restore the log returns via inverse sigmoid
        vid_pred_flat = vid_pred.view(-1)
        vid_pred_flat = vid_pred_flat.tolist()
        vid_pred_samples_append = [sum(x) for x in zip(vid_pred_samples_append, vid_pred_flat)] #Add all 100 video prediction sample results (log returns) and compute the average
      vid_pred_samples_append = [x / nfore for x in vid_pred_samples_append]
    vid_pred_seeds_append = [sum(x) for x in zip(vid_pred_seeds_append, vid_pred_samples_append)]
  vid_pred_seeds_append = [x / nseeds for x in vid_pred_seeds_append]

  #Prepare test data in a list

  number_of_assets = 9
  timesteps_forecast = 10

  test_data = []
  for asset in range(number_of_assets):
    for timestep in range(timesteps_forecast):
      test_data.append(stock_prices.iloc[751 + timestep, asset]) #Appending starts at Mo

  log_forecast = []
  index_shift = 0
  for asset in range(number_of_assets):
    log_cumulative_asset = log_stock_prices.iloc[750, asset] #Computation starts from ln price of Fr (add up the log returns)
    for timestep in range (timesteps_forecast):
      log_cumulative_asset = log_cumulative_asset + vid_pred_seeds_append[timestep + index_shift]
      log_forecast.append(log_cumulative_asset)
    index_shift = index_shift + 10
  forecast = np.exp(log_forecast)
  print(forecast, 'Forecasts of adjusted closing prices')
  print(list_of_random_seeds, 'Random seeds:')
  list_of_random_seeds = [] #RESET LISTS, SO THAT SEEDS AND RESULTS DON'T ACCUMULATE
  vid_pred_samples_append = [0.00] * 90 #RESET LISTS, SO THAT SEEDS AND RESULTS DON'T ACCUMULATE
  vid_pred_seeds_append = [0.00] * 90 #RESET LISTS, SO THAT SEEDS AND RESULTS DON'T ACCUMULATE

  #Compute the MAPE

  MAPE_list = []
  total_errors = number_of_assets * timesteps_forecast
  for error in range(total_errors):
    MAPE = np.abs(test_data[error] - forecast[error]) / test_data[error]
    MAPE_list.append(MAPE)
  print(MAPE_list, 'MAPE')
  MAPE_assets = []
  index_shift = 0
  for asset in range(number_of_assets):
    MAPE_asset = mean_absolute_percentage_error(test_data[index_shift : index_shift + 10], forecast[index_shift : index_shift + 10])
    MAPE_assets.append(MAPE_asset)
    index_shift = index_shift + 10
  print(MAPE_assets, 'average MAPE (over ten timesteps/per asset)')

  #Compute the RRMSE

  RRMSE_assets = []
  index_shift = 0
  for asset in range(number_of_assets):
    RRMSE_asset_sum = 0.00
    for timestep in range(timesteps_forecast):
      RRMSE_asset_sum = RRMSE_asset_sum + (((forecast[timestep + index_shift] - test_data[timestep + index_shift])
                                                     / test_data[timestep + index_shift]) ** 2)
    RRMSE_asset = np.sqrt(RRMSE_asset_sum / timesteps_forecast)
    RRMSE_assets.append(RRMSE_asset)
    index_shift = index_shift + 10
  print(RRMSE_assets, 'Individual RRMSE (per asset)')
  RRMSE_assets = sum(RRMSE_assets)
  RRMSE_assets = RRMSE_assets / number_of_assets
  print(RRMSE_assets, 'RRMSE')

  #Compute the price movement direction metric

  price_movement_direction_assets = []
  weight_lambda = 0.60
  index_shift = 0
  for asset in range(number_of_assets):
    price_movement_direction_assets_sum = 0.00
    if (((forecast[index_shift] - stock_prices.iloc[750, asset]) / (test_data[index_shift] - stock_prices.iloc[750, asset])) > 0):
      price_movement_direction_assets_sum = price_movement_direction_assets_sum + (1 * (weight_lambda ** 0))
    for timestep in range(timesteps_forecast - 1):
      if (((forecast[timestep + 1 + index_shift] - forecast[timestep + index_shift]) / (test_data[timestep + 1 + index_shift] - test_data[timestep + index_shift])) > 0):
        price_movement_direction_assets_sum = price_movement_direction_assets_sum + (1 * (weight_lambda ** (timestep + 1)))
    price_movement_direction_assets.append(price_movement_direction_assets_sum)
    index_shift = index_shift + 10
  print(price_movement_direction_assets, 'Individual price movement accuracy metric (per asset)')
  price_movement_direction_assets = sum(price_movement_direction_assets)
  print(price_movement_direction_assets, 'Price movement accuracy metric')

#Trigger main function

main()
