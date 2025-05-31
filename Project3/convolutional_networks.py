"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from p3_utils import softmax_loss, Solver, stacked_dilate2d
from fully_connected_networks import Linear_ReLU, Linear, adam, ReLU
from math import floor

#Original Author: Justin C. Johnson https://web.eecs.umich.edu/~justincj/
#Modified: Jonathan Gryak, Spring 2025

def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')

class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        N, C, H, W =  x.shape
        F, C, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        padded_x = torch.nn.functional.pad(x, [pad] * 4)
        out_H, out_W = (H + 2 * pad - HH) // stride + 1, (W + 2 * pad - WW) // stride + 1
        out = torch.zeros((N, F, out_H, out_W), dtype=x.dtype, device=x.device)

        for i in range(N):
          for j in range(F):
            out[i, j] = b[j]
            for k in range(out_H):
              start_h = k * stride
              end_h = start_h + HH
              for l in range(out_W):
                start_w = l * stride
                end_w = start_w + WW
                window = padded_x[i, :, start_h: end_h, start_w: end_w]
                out[i, j, k, l] += torch.sum(window * w[j])
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # The convolutional backward pass.                            #
        # IMPLEMENTED by J. Gryak - DO NOT MODIFY                     #
        ###############################################################
        # Replace "pass" statement with your code
        x, w, b, conv_param=cache
        N,C,H,W=x.shape
        #F,_,HH,WW=w.shape
        F,_,FH,FW=w.shape
        _,_,OH,OW=dout.shape
        P = conv_param['pad']
        S = conv_param['stride']
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)
        
        padded_x = torch.nn.functional.pad(x,[P]*4)
        _, _, padded_H, padded_W = padded_x.shape
        x_col = torch.zeros((C * FH * FW, H * W),device=x.device,dtype=x.dtype)
        w_row = w.reshape(F, C * FH * FW)

        for i in range(N):
          #dx, modified from original source by  D. Nguyen
          curr_dout = dout[i, :, :, :].reshape(F, OH * OW)
          curr_out = torch.mm(w_row.T, curr_dout)
          curr_dpx = torch.zeros(padded_x.shape[1:],device=x.device,dtype=x.dtype)
          c = 0
          for j in range(0, padded_H - FH + 1, S):
            for k in range(0, padded_W - FW + 1, S):
              curr_dpx[:, j:j+FH, k:k+FW] += curr_out[:, c].reshape(C, FH, FW)
              x_col[:, c] = padded_x[i, :, j:j+FH, k:k+FW].reshape(C * FH * FW) 
              c += 1
          dx[i] = curr_dpx[:, P:-P, P:-P]

          #dw
          d_dout=stacked_dilate2d(dout[i,:,:,:],S)
          _,ddH, ddW=d_dout.shape
          for f in range(F):
              for c in range(C):
                for j in range (FH):
                  for k in range(FW):
                    dw[f,c,j,k]+=(d_dout[f,:,:]*padded_x[i,c,j:j+ddH,k:k+ddW]).sum()

        #db
        db = dout.sum((0,2,3))

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################  
        # Replace "pass" statement with your code
        N, C, H, W = x.shape 
        HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        out_h, out_w = (H - HH) // stride + 1, (W - WW) // stride + 1
        out = torch.zeros((N, C, out_h, out_w), dtype=x.dtype, device=x.device)

        for i in range(N):
          for j in range(C):
            for k in range(out_h):
              h_start = k * stride
              h_end = h_start + HH
              for l in range(out_w):
                  w_start = l * stride
                  w_end = w_start + WW
                  window = x[i, j, h_start: h_end, w_start: w_end]
                  out[i, j, k, l] = torch.max(window)
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # The max-pooling backward pass                                     #
        # IMPLEMENTED by J. Gryak - DO NOT MODIFY                           #
        #####################################################################
        # Replace "pass" statement with your code
        x, pool_param = cache
        HH=pool_param['pool_height']
        WW=pool_param['pool_width']
        S=pool_param['stride']
        N, C, H, W = x.shape
        _,_,OH, OW = dout.shape 
        dx = torch.zeros_like(x)

        for i in range(N):
          curr_dout = dout[i, :].reshape(C, OH * OW)
          c = 0
          for j in range(0, H - HH + 1, S):
            for k in range(0, W - WW + 1, S): 
              curr_region = x[i, :, j:j+HH, k:k+WW].reshape(C, HH * WW)
              curr_max_idx = torch.argmax(curr_region, dim=1)
              curr_dout_region = curr_dout[:, c]
              curr_dpooling = torch.zeros_like(curr_region)
              curr_dpooling[torch.arange(C), curr_max_idx] = curr_dout_region
              dx[i, :, j:j+HH, k:k+WW] = curr_dpooling.reshape(C, HH, WW)
              c += 1
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in the dictionary self.params. Store weights and  #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        C, H, W = input_dims
        self.params['W1'] = torch.randn(num_filters, C, filter_size, filter_size, dtype=dtype, device=device) * weight_scale
        self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)
        pool_h = pool_w = pool_stride = 2
        HP, WP = (H - pool_h) // pool_stride + 1, (W - pool_w) // pool_stride + 1
        self.params['W2'] = torch.randn(num_filters * HP * WP, hidden_dim, dtype=dtype, device=device) * weight_scale
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W3'] = torch.randn(hidden_dim, num_classes, dtype=dtype, device=device) * weight_scale
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable. You may use torch.nn.functional.softmax to        #
        # compute the scores.                                                #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above.                                                             #
        ######################################################################
        # Replace "pass" statement with your code
        out, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = Linear_ReLU.forward(out, W2, b2)
        scores, cache3 = Linear.forward(out, W3, b3)
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2) + torch.sum(W3 ** 2))
        dout, grads['W3'], grads['b3'] = Linear.backward(dscores, cache3)
        dout, grads['W2'], grads['b2'] = Linear_ReLU.backward(dout, cache2)
        dout, grads['W1'], grads['b1'] = Conv_ReLU_Pool.backward(dout, cache1)
        grads['W3'] += 2 * self.reg * W3
        grads['W2'] += 2 * self.reg * W2
        grads['W1'] += 2 * self.reg * W1
        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    , a ReLU nonlinearity, and an optional pooling layer. After L-1 such macro 
    layers, a single fully-connected layer is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights  #
        # and biases should be stored in the dictionary self.params.        #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        #####################################################################
        # Replace "pass" statement with your code
        C, H, W = input_dims
        prev = C
        for i, num_filter in enumerate(num_filters):
          if weight_scale == 'kaiming':
            self.params[f'W{i + 1}'] = kaiming_initializer(Din=prev, Dout=num_filter, K=3, relu=True, device=device, dtype=dtype)
          else:
            self.params[f'W{i + 1}'] = torch.randn(num_filter, prev, 3, 3, dtype=dtype, device=device) * weight_scale
          self.params[f'b{i + 1}'] = torch.zeros(num_filter, dtype=dtype, device=device)
          prev = num_filter
          if i in max_pools:
            W = W // 2
            H = H // 2
        fcl_input_dim = num_filters[-1] * H * W
        if weight_scale == 'kaiming':
          self.params[f'W{self.num_layers}'] = kaiming_initializer(Din=fcl_input_dim, Dout=num_classes, K=None, relu=False, device=device, dtype=dtype)
        else:
          self.params[f'W{self.num_layers}'] = torch.randn(fcl_input_dim, num_classes, dtype=dtype, device=device) * weight_scale
        self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=dtype, device=device)
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # Check that we got the right number of parameters
        params_per_macro_layer = 2  # weight and bias
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        out = X
        caches = []
        for i in range(1, self.num_layers):
          w = self.params[f'W{i}']
          b = self.params[f'b{i}']
          out, curr_cache = Conv_ReLU.forward(out, w, b, conv_param)
          if (i - 1) in self.max_pools:
            out, pool_cache = FastMaxPool.forward(out, pool_param)
            curr_cache = (curr_cache, pool_cache)
          caches.append(curr_cache)
        out_flattened = out.reshape(out.shape[0], -1)
        scores, cache_l = Linear.forward(out_flattened, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
        caches.append(cache_l)
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
          reg_loss += torch.sum(self.params[f'W{i}'] ** 2)
        loss += self.reg * reg_loss

        dout_flattened, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = Linear.backward(dscores, caches[-1])
        dout = dout_flattened.reshape(out.shape)
        for i in range(self.num_layers - 1, 0, -1):
          curr_cache = caches[i - 1]
          if (i - 1) in self.max_pools:
            conv_cache, pool_cache = curr_cache
            dout = FastMaxPool.backward(dout, pool_cache)
            dout, dw, db = Conv_ReLU.backward(dout, conv_cache)
          else:
            dout, dw, db = Conv_ReLU.backward(dout, curr_cache)
          grads[f'W{i}'] = dw + 2 * self.reg * self.params[f'W{i}']
          grads[f'b{i}'] = db
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3 # Experiment with this!
    learning_rate = 1e-5 # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 1e-1
    learning_rate = 5e-3
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    model = DeepConvNet(
      input_dims=data_dict['X_train'].shape[1:],
      num_filters=[32, 64, 128], 
      max_pools = [0, 1],
      num_classes=10,
      weight_scale='kaiming',
      reg=6e-4,
      dtype=dtype,
      device=device
    )
    solver = Solver(
      model,
      data_dict,
      num_epochs=12,
      batch_size=128,
      update_rule=adam,
      optim_config={'learning_rate': 2e-3},
      print_every=50,
      device=device
    )
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver

def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        from math import sqrt
        fan_in = Din
        weight_scale = sqrt(gain / fan_in)
        weight = torch.randn(Din, Dout, device=device, dtype=dtype) * weight_scale
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        from math import sqrt
        fan_in = Din * K * K
        weight_scale = sqrt(gain / fan_in)
        weight = torch.randn(Dout, Din, K, K, device=device, dtype=dtype) * weight_scale
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db

