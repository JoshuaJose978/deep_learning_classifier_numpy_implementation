import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
import json
from copy import deepcopy
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class Layer:
    def __init__(self, name=None):
        self.name = name
        self.trainable = True
        
    def forward(self, inputs, training=True):
        raise NotImplementedError
        
    def backward(self, grad_output):
        raise NotImplementedError
        
    def get_parameters(self):
        return []
        
    def get_gradients(self):
        return []
    
    def get_config(self):
        """Return configuration of the layer for saving/loading"""
        return {"name": self.name, "trainable": self.trainable}


class Dense(Layer):
    def __init__(self, input_dim, output_dim, name=None):
        super().__init__(name=name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # He initialization
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases = np.zeros(output_dim)
        
        # Gradient placeholders
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.biases)
        self.inputs = None
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
        
    def backward(self, grad_output):
        # Calculate gradients
        self.dW = np.dot(self.inputs.T, grad_output)
        self.db = np.sum(grad_output, axis=0)
        
        # Return gradient for next layer
        return np.dot(grad_output, self.weights.T)
        
    def get_parameters(self):
        return [self.weights, self.biases]
        
    def get_gradients(self):
        return [self.dW, self.db]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "class": "Dense",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        })
        return config


class Dropout(Layer):
    def __init__(self, rate=0.5, name=None):
        super().__init__(name=name)
        self.rate = rate
        self.mask = None
        self.trainable = False
        
    def forward(self, inputs, training=True):
        # Make sure dropout is only applied during training
        if not training:
            return inputs
            
        # Generate mask with proper scaling
        self.mask = np.random.binomial(1, 1-self.rate, size=inputs.shape) / (1-self.rate)
        return inputs * self.mask
        
    def backward(self, grad_output):
        return grad_output * self.mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "class": "Dropout",
            "rate": self.rate
        })
        return config


class BatchNormalization(Layer):
    def __init__(self, input_dim, epsilon=1e-5, momentum=0.9, name=None, axis=1):
        super().__init__(name=name)
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        self.epsilon = epsilon
        self.momentum = momentum
        self.axis = axis  # Axis along which to normalize (1 for FC, (0,2,3) for conv)
        self.input_dim = input_dim
        
        # Moving statistics for inference
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
        # Cache for backpropagation
        self.cache = None
        
        # Gradients
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        
    def forward(self, inputs, training=True):
        # Handle multi-dimensional inputs (e.g., for convolutional layers)
        input_shape = inputs.shape
        ndim = len(input_shape)
        
        # Determine axes to normalize over
        if ndim == 2:  # Fully-connected layer
            # Normalize over batch dimension
            reduce_axes = 0
            broadcast_shape = (1, input_shape[1])
        elif ndim == 4:  # Conv layer with shape (batch, channels, height, width)
            # Normalize over batch, height, and width dimensions
            if self.axis == 1:  # Channel axis is 1 (channels first)
                reduce_axes = (0, 2, 3)
                broadcast_shape = (1, input_shape[1], 1, 1)
            else:  # Channel axis is 3 (channels last)
                reduce_axes = (0, 1, 2)
                broadcast_shape = (1, 1, 1, input_shape[3])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
            
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(inputs, axis=reduce_axes, keepdims=True)
            batch_var = np.var(inputs, axis=reduce_axes, keepdims=True)
            
            # Update running statistics
            if ndim == 2:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.reshape(-1)
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.reshape(-1)
            else:
                # For conv layers, reshape back to 1D for running stats
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.reshape(-1)
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.reshape(-1)
            
            # Normalize
            x_normalized = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Cache for backpropagation
            self.cache = (x_normalized, batch_mean, batch_var, inputs, reduce_axes, broadcast_shape)
        else:
            # Use running statistics for inference
            if ndim == 2:
                batch_mean = self.running_mean.reshape(1, -1)
                batch_var = self.running_var.reshape(1, -1)
            else:
                # Reshape running stats for conv layers
                batch_mean = self.running_mean.reshape(broadcast_shape)
                batch_var = self.running_var.reshape(broadcast_shape)
                
            x_normalized = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
        # Scale and shift
        if ndim == 2:
            return self.gamma * x_normalized + self.beta
        else:
            # Reshape gamma and beta for conv layers
            gamma_reshaped = self.gamma.reshape(broadcast_shape)
            beta_reshaped = self.beta.reshape(broadcast_shape)
            return gamma_reshaped * x_normalized + beta_reshaped
            
    def backward(self, grad_output):
        # Retrieve cache
        x_normalized, batch_mean, batch_var, inputs, reduce_axes, broadcast_shape = self.cache
        
        # Get dimensions
        if len(inputs.shape) == 2:
            N = inputs.shape[0]
            self.dgamma = np.sum(grad_output * x_normalized, axis=0)
            self.dbeta = np.sum(grad_output, axis=0)
            
            # Gradient with respect to normalized input
            dx_normalized = grad_output * self.gamma
            
            # Gradient with respect to input
            dvar = np.sum(dx_normalized * (inputs - batch_mean) * -0.5 * np.power(batch_var + self.epsilon, -1.5), axis=0)
            dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + self.epsilon), axis=0) + \
                    dvar * np.sum(-2 * (inputs - batch_mean), axis=0) / N
            
            dx = dx_normalized / np.sqrt(batch_var + self.epsilon) + \
                 dvar * 2 * (inputs - batch_mean) / N + \
                 dmean / N
        else:
            # For conv layers
            # Calculate N as the product of dimensions we're normalizing over
            N = np.prod([inputs.shape[i] for i in range(len(inputs.shape)) if i in reduce_axes])
            
            # Compute gradients for gamma and beta
            self.dgamma = np.sum(grad_output * x_normalized, axis=reduce_axes)
            self.dbeta = np.sum(grad_output, axis=reduce_axes)
            
            # Reshape gamma for multiplication
            gamma_reshaped = self.gamma.reshape(broadcast_shape)
            
            # Gradient with respect to normalized input
            dx_normalized = grad_output * gamma_reshaped
            
            # Gradient with respect to variance
            dvar = np.sum(dx_normalized * (inputs - batch_mean) * -0.5 * np.power(batch_var + self.epsilon, -1.5), 
                          axis=reduce_axes, keepdims=True)
            
            # Gradient with respect to mean
            dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + self.epsilon), 
                           axis=reduce_axes, keepdims=True)
            dmean += dvar * np.sum(-2 * (inputs - batch_mean), axis=reduce_axes, keepdims=True) / N
            
            # Gradient with respect to input
            dx = dx_normalized / np.sqrt(batch_var + self.epsilon)
            dx += dvar * 2 * (inputs - batch_mean) / N
            dx += dmean / N
            
        return dx
        
    def get_parameters(self):
        return [self.gamma, self.beta]
        
    def get_gradients(self):
        return [self.dgamma, self.dbeta]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "class": "BatchNormalization",
            "input_dim": self.input_dim,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "axis": self.axis
        })
        return config


class Activation(Layer):
    def __init__(self, activation_fn, name=None):
        super().__init__(name=name)
        self.activation_fn = activation_fn
        self.trainable = False
        self.inputs = None
        
    def forward(self, inputs, training=True):
        self.inputs = inputs
        return self.activation_fn.forward(inputs)
        
    def backward(self, grad_output):
        return self.activation_fn.backward(grad_output, self.inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "class": "Activation",
            "activation": self.activation_fn.__class__.__name__
        })
        return config


class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)
        
    def backward(self, grad_output, inputs):
        return grad_output * (inputs > 0)


class Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-np.clip(inputs, -15, 15)))
        
    def backward(self, grad_output, inputs):
        sigmoid_output = self.forward(inputs)
        return grad_output * sigmoid_output * (1 - sigmoid_output)


class Softmax:
    def forward(self, inputs):
        # Shift inputs for numerical stability
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
        
    def backward(self, grad_output, inputs):
        # Proper softmax gradient calculation
        n_samples = inputs.shape[0]
        jacobians = np.zeros((n_samples, inputs.shape[1], inputs.shape[1]))
        
        for i in range(n_samples):
            softmax_output = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
            jacobians[i] = jacobian
            
        # For each sample, compute the dot product of gradient with Jacobian
        dx = np.zeros_like(grad_output)
        for i in range(n_samples):
            dx[i] = np.dot(grad_output[i], jacobians[i])
            
        return dx


class Loss:
    def calculate(self, predictions, targets):
        raise NotImplementedError
        
    def gradient(self, predictions, targets):
        raise NotImplementedError


class CategoricalCrossentropy(Loss):
    def calculate(self, predictions, targets):
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # If targets are one-hot encoded
        if len(targets.shape) == 2:
            return -np.mean(np.sum(targets * np.log(predictions), axis=1))
        # If targets are sparse (class indices)
        else:
            return -np.mean(np.log(predictions[np.arange(len(predictions)), targets]))
            
    def gradient(self, predictions, targets):
        # Clip predictions to avoid division by zero
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # If targets are one-hot encoded
        if len(targets.shape) == 2:
            return -targets / predictions / len(predictions)
        # If targets are sparse (class indices)
        else:
            grad = np.zeros_like(predictions)
            grad[np.arange(len(predictions)), targets] = -1 / predictions[np.arange(len(predictions)), targets]
            return grad / len(predictions)


class BinaryCrossentropy(Loss):
    def calculate(self, predictions, targets):
        # Clip predictions to avoid log(0) or log(1)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        
    def gradient(self, predictions, targets):
        # Clip predictions to avoid division by zero
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -(targets / predictions - (1 - targets) / (1 - predictions)) / len(predictions)


class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update_params(self, layers):
        raise NotImplementedError
    
    def get_config(self):
        return {"learning_rate": self.learning_rate}


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = []
        
    def update_params(self, layers):
        # Initialize velocity if it's the first update
        if not self.velocity:
            for layer in layers:
                if layer.trainable:
                    layer_velocity = []
                    for param in layer.get_parameters():
                        layer_velocity.append(np.zeros_like(param))
                    self.velocity.append(layer_velocity)
        
        # Update parameters with momentum
        layer_idx = 0
        for layer in layers:
            if layer.trainable:
                for param_idx, (param, grad) in enumerate(zip(layer.get_parameters(), layer.get_gradients())):
                    # Update velocity
                    self.velocity[layer_idx][param_idx] = self.momentum * self.velocity[layer_idx][param_idx] - self.learning_rate * grad
                    
                    # Update parameter
                    param += self.velocity[layer_idx][param_idx]
                layer_idx += 1
    
    def get_config(self):
        config = super().get_config()
        config.update({"momentum": self.momentum})
        return config


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []  # First moment estimates
        self.v = []  # Second moment estimates
        self.t = 0   # Timestep
        
    def update_params(self, layers):
        # Initialize moments if it's the first update
        if not self.m:
            for layer in layers:
                if layer.trainable:
                    layer_m = []
                    layer_v = []
                    for param in layer.get_parameters():
                        layer_m.append(np.zeros_like(param))
                        layer_v.append(np.zeros_like(param))
                    self.m.append(layer_m)
                    self.v.append(layer_v)
        
        # Increment timestep
        self.t += 1
        
        # Update parameters with Adam
        layer_idx = 0
        for layer in layers:
            if layer.trainable:
                for param_idx, (param, grad) in enumerate(zip(layer.get_parameters(), layer.get_gradients())):
                    # Update biased first moment estimate
                    self.m[layer_idx][param_idx] = self.beta1 * self.m[layer_idx][param_idx] + (1 - self.beta1) * grad
                    
                    # Update biased second raw moment estimate
                    self.v[layer_idx][param_idx] = self.beta2 * self.v[layer_idx][param_idx] + (1 - self.beta2) * (grad ** 2)
                    
                    # Compute bias-corrected first moment estimate
                    m_corrected = self.m[layer_idx][param_idx] / (1 - self.beta1 ** self.t)
                    
                    # Compute bias-corrected second raw moment estimate
                    v_corrected = self.v[layer_idx][param_idx] / (1 - self.beta2 ** self.t)
                    
                    # Update parameter
                    param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
                layer_idx += 1
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        })
        return config


class Callback:
    def on_epoch_begin(self, epoch, logs=None):
        pass
        
    def on_epoch_end(self, epoch, logs=None):
        pass
        
    def on_batch_begin(self, batch, logs=None):
        pass
        
    def on_batch_end(self, batch, logs=None):
        pass
        
    def on_train_begin(self, logs=None):
        pass
        
    def on_train_end(self, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=5, min_delta=0, mode='min'):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.model = None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        # Make a deep copy of the initial weights
        self.best_weights = self.model.get_weights()
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is None:
            return
            
        if self.mode == 'min':
            improvement = self.best_value - current > self.min_delta
        else:
            improvement = current - self.best_value > self.min_delta
            
        if improvement:
            self.best_value = current
            self.wait = 0
            # Save a deep copy of the weights to avoid any potential reference issues
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                # Restore the best weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                print(f"\nEarly stopping triggered at epoch {epoch+1}")


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.model = None
        self.save_weights_only = save_weights_only
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is None:
            return
            
        if self.save_best_only:
            if self.mode == 'min':
                improvement = self.best_value > current
            else:
                improvement = self.best_value < current
                
            if improvement:
                self.best_value = current
                self.save_model(epoch, logs)
        else:
            self.save_model(epoch, logs)
            
    def save_model(self, epoch, logs):
        # Format filepath with epoch and metrics
        formatted_filepath = self.filepath.format(epoch=epoch+1, **logs)
        
        try:
            if self.save_weights_only:
                # Save model weights
                self.model.save_weights(formatted_filepath)
            else:
                # Save full model
                self.model.save(formatted_filepath)
                
            print(f"\nModel saved to {formatted_filepath}")
        except Exception as e:
            print(f"\nError saving model to {formatted_filepath}: {str(e)}")


class LearningRateScheduler(Callback):
    def __init__(self, schedule, verbose=0):
        self.schedule = schedule
        self.verbose = verbose
        self.model = None
        
    def on_epoch_begin(self, epoch, logs=None):
        # Get the learning rate for this epoch
        lr = self.schedule(epoch)
        
        # Set the learning rate in optimizer
        self.model.optimizer.learning_rate = lr
        
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: Learning rate set to {lr}")


class TensorBoard(Callback):
    def __init__(self, log_dir='logs', histogram_freq=0, update_freq='epoch'):
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.update_freq = update_freq
        self.epoch = 0
        self.model = None
        self.writer_path = None
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
    def on_train_begin(self, logs=None):
        # Create writer for TensorBoard
        self.writer_path = os.path.join(self.log_dir, f"run_{int(time.time())}")
        os.makedirs(self.writer_path, exist_ok=True)
        
        # Create histogram directory at the beginning if needed
        if self.histogram_freq > 0:
            hist_dir = os.path.join(self.writer_path, "histograms")
            os.makedirs(hist_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        
        # Write epoch metrics
        if logs:
            metrics_file = os.path.join(self.writer_path, f"metrics.json")
            
            # Read existing metrics if the file exists
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            else:
                metrics_data = []
                
            # Add current epoch metrics
            metrics_data.append({
                'epoch': epoch + 1,
                **logs
            })
            
            # Write updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        # Write histograms
        if self.histogram_freq > 0 and (epoch + 1) % self.histogram_freq == 0:
            self.write_histograms(epoch)
            
    def write_histograms(self, epoch):
        # Create histogram directory for this epoch
        hist_dir = os.path.join(self.writer_path, "histograms", f"epoch_{epoch+1}")
        os.makedirs(hist_dir, exist_ok=True)
        
        # Save histograms for each trainable layer
        for layer in self.model.layers:
            if layer.trainable and layer.name:
                for i, param in enumerate(layer.get_parameters()):
                    param_name = f"weights" if i == 0 else f"biases"
                    
                    try:
                        # Create histogram plot
                        plt.figure(figsize=(8, 6))
                        plt.hist(param.flatten(), bins=50)
                        plt.title(f"{layer.name} - {param_name}")
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                        
                        # Save plot
                        plt.savefig(os.path.join(hist_dir, f"{layer.name}_{param_name}.png"))
                        plt.close()
                    except Exception as e:
                        print(f"Error creating histogram for {layer.name}_{param_name}: {str(e)}")
                    
    def on_train_end(self, logs=None):
        # Create final plots
        self.create_training_plots()
        
    def create_training_plots(self):
        # Read metrics data
        metrics_file = os.path.join(self.writer_path, f"metrics.json")
        if not os.path.exists(metrics_file):
            return
            
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)
            
        if not metrics_data:
            return
            
        # Extract metrics
        epochs = [data['epoch'] for data in metrics_data]
        available_metrics = list(metrics_data[0].keys())
        available_metrics.remove('epoch')
        
        # Create plots directory
        plots_dir = os.path.join(self.writer_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot each metric
        for metric in available_metrics:
            metric_values = [data[metric] for data in metrics_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, metric_values, 'b-')
            plt.title(f'Training {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        # Evaluate the model
        test_metrics = model.evaluate(X_test, y_test)
        print("\nTest metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Plot decision boundary
        plt.figure(figsize=(10, 8))
        
        # Create a mesh grid
        h = 0.05  # Step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Standardize mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_std = scaler.transform(mesh_points)
        
        # Predict on mesh grid
        Z = model.predict(mesh_points_std)
        Z = (Z > 0.5).astype(int)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        
        # Plot data points
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig('decision_boundary.png')
        plt.show()
        
        # Save the final model
        model.save('final_model.json')
        print("Model saved to final_model.json")
        
        # Demonstrate loading a model and making predictions
        print("\nLoading model and comparing predictions...")
        try:
            # Load the model
            new_model = NeuralNetwork.load_model('final_model.json')
            
            if new_model is not None:
                # Make predictions with loaded model
                new_predictions = new_model.predict(X_test)
                new_pred_binary = (new_predictions > 0.5).astype(int)
                
                # Check if predictions match
                match_rate = np.mean(new_pred_binary == y_pred_binary) * 100
                print(f"Predictions from loaded model match original model: {match_rate:.2f}%")
            else:
                print("Failed to load model")
                
        except Exception as e:
            print(f"Error testing model loading: {str(e)}")
            
        # Test the copy/modify architecture pattern
        print("\nTesting architecture modification...")
        try:
            # Create a new model with a slightly different architecture
            modified_model = NeuralNetwork()
            
            # Add layers
            modified_model.add(Dense(2, 64, name="input_layer"))  # Different number of neurons
            modified_model.add(Activation(ReLU(), name="relu1"))
            modified_model.add(BatchNormalization(64, name="batchnorm1"))
            modified_model.add(Dense(64, 32, name="hidden_layer1"))  # Different number of neurons
            modified_model.add(Activation(ReLU(), name="relu2"))
            modified_model.add(Dropout(0.3, name="dropout1"))
            modified_model.add(Dense(32, 16, name="hidden_layer2"))  # Different number of neurons
            modified_model.add(Activation(ReLU(), name="relu3"))
            modified_model.add(Dense(16, 1, name="output_layer"))
            modified_model.add(Activation(Sigmoid(), name="sigmoid_output"))
            
            # Compile the model
            modified_model.compile(
                loss=BinaryCrossentropy(),
                optimizer=Adam(learning_rate=0.01)
            )
            
            # Try to load weights selectively (should fail for mismatched layers)
            success = modified_model.load_weights('final_model.json')
            print(f"Partial weight loading {'succeeded' if success else 'failed'} as expected")
            
        except Exception as e:
            print(f"Error in architecture modification test: {str(e)}")
            
    except Exception as e:
        print(f"Error during model training or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, f"{metric}.png"))
            plt.close()
            
        # If both loss and val_loss exist, plot them together
        if 'loss' in available_metrics and 'val_loss' in available_metrics:
            loss_values = [data['loss'] for data in metrics_data]
            val_loss_values = [data['val_loss'] for data in metrics_data]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, loss_values, 'b-', label='Training Loss')
            plt.plot(epochs, val_loss_values, 'r-', label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, "loss_comparison.png"))
            plt.close()


class ProgressBar(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.model = None
        
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.batch_times = []
        print(f"Epoch {epoch+1}/{self.total_epochs}")
        
    def on_batch_end(self, batch, logs=None):
        self.batch_times.append(time.time() - self.start_time)
        self.start_time = time.time()
        
        # Calculate average time per batch
        avg_time = np.mean(self.batch_times)
        
        # Print progress
        progress = min(20, int(20 * (batch + 1) / self.model.steps_per_epoch))
        bar = "[" + "=" * progress + " " * (20 - progress) + "]"
        
        # Print stats
        stats = []
        for key, value in logs.items():
            stats.append(f"{key}: {value:.4f}")
        
        # Print line with carriage return to update the same line
        print(f"\r{batch+1}/{self.model.steps_per_epoch} {bar} - {' - '.join(stats)} - {avg_time:.3f}s/batch", end="")
        
    def on_epoch_end(self, epoch, logs=None):
        # Print end of epoch with newline
        stats = []
        for key, value in logs.items():
            stats.append(f"{key}: {value:.4f}")
        print(f"\nEpoch {epoch+1}/{self.total_epochs} completed - {' - '.join(stats)}")


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.stop_training = False
        self.steps_per_epoch = 0
        self._compiled = False
        
    def add(self, layer):
        """Add a layer to the network"""
        if layer.name is None:
            layer.name = f"layer_{len(self.layers)}"
        self.layers.append(layer)
        
    def compile(self, loss, optimizer):
        """Configure the model for training"""
        self.loss_function = loss
        self.optimizer = optimizer
        self._compiled = True
        
    def forward(self, X, training=True):
        """Forward pass through the network"""
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output
        
    def backward(self, grad):
        """Backward pass through the network"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
        
    def train_on_batch(self, X, y):
        """Train the model on a single batch"""
        # Forward pass
        predictions = self.forward(X, training=True)
        
        # Calculate loss
        loss = self.loss_function.calculate(predictions, y)
        
        # Calculate initial gradient
        grad = self.loss_function.gradient(predictions, y)
        
        # Backward pass
        self.backward(grad)
        
        # Update parameters
        self.optimizer.update_params(self.layers)
        
        # Calculate metrics for the batch
        batch_metrics = self._calculate_metrics(predictions, y)
            
        return loss, batch_metrics
    
    def _calculate_metrics(self, predictions, y):
        """Calculate various metrics based on predictions and targets"""
        metrics = {}
        
        # For binary classification
        if hasattr(predictions, 'shape') and len(predictions.shape) <= 2 and (predictions.shape[1] if len(predictions.shape) > 1 else 1) == 1:
            # Flatten predictions and targets for binary metrics
            flat_preds = predictions.flatten()
            flat_y = y.flatten() if len(y.shape) > 1 else y
            
            # Binarize predictions for metrics that need binary values
            binary_preds = (flat_preds > 0.5).astype(int)
            
            # Calculate accuracy
            metrics['accuracy'] = np.mean(binary_preds == flat_y)
            
            try:
                # Calculate precision, recall, F1
                metrics['precision'] = precision_score(flat_y, binary_preds)
                metrics['recall'] = recall_score(flat_y, binary_preds)
                metrics['f1'] = f1_score(flat_y, binary_preds)
                
                # Calculate ROC AUC if possible (can fail on batches with one class)
                if len(np.unique(flat_y)) > 1:
                    metrics['auc'] = roc_auc_score(flat_y, flat_preds)
            except Exception as e:
                # Some metrics can fail on certain batches (e.g., if a batch has only one class)
                # Just log the error and continue
                pass
                
        # For multi-class classification
        elif hasattr(predictions, 'shape') and len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Get predicted classes
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1) if len(y.shape) > 1 else y
            
            # Calculate accuracy
            metrics['accuracy'] = np.mean(pred_classes == true_classes)
            
        return metrics
        
    def evaluate(self, X, y, batch_size=32):
        """Evaluate the model on test data"""
        if not self._compiled:
            raise ValueError("Model must be compiled before evaluation.")
            
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Forward pass
            predictions = self.forward(X_batch, training=False)
            all_predictions.append(predictions)
            all_targets.append(y_batch)
            
            # Calculate loss
            loss = self.loss_function.calculate(predictions, y_batch)
            total_loss += loss * (end_idx - start_idx)
            
        # Concatenate predictions and targets
        all_predictions = np.vstack(all_predictions)
        
        # Handle different shapes of targets
        if isinstance(all_targets[0], np.ndarray) and len(all_targets[0].shape) > 1:
            all_targets = np.vstack(all_targets)
        else:
            all_targets = np.concatenate(all_targets)
            
        # Calculate average loss
        avg_loss = total_loss / n_samples
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
        
    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None, callbacks=None, verbose=1):
        """Train the model on data"""
        if not self._compiled:
            raise ValueError("Model must be compiled before training.")
            
        n_samples = X_train.shape[0]
        self.steps_per_epoch = int(np.ceil(n_samples / batch_size))
        
        # Initialize callbacks
        callbacks = callbacks or []
        for callback in callbacks:
            callback.model = self
            
        # Call on_train_begin for all callbacks
        logs = {}
        for callback in callbacks:
            callback.on_train_begin(logs)
            
        # Training loop
        history = {'loss': []}
        
        # Initialize history with metrics
        batch_metrics = self._calculate_metrics(self.forward(X_train[:min(batch_size, n_samples)]), 
                                             y_train[:min(batch_size, n_samples)])
        for metric in batch_metrics:
            history[metric] = []
            
        if validation_data:
            history['val_loss'] = []
            # Add validation metrics to history
            val_metrics = self.evaluate(*validation_data, batch_size=batch_size)
            for metric in val_metrics:
                if f'val_{metric}' not in history:
                    history[f'val_{metric}'] = []
            
        for epoch in range(epochs):
            if self.stop_training:
                break
                
            # Call on_epoch_begin for all callbacks
            logs = {}
            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)
                
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            epoch_metrics = {}
            
            for batch in range(self.steps_per_epoch):
                # Call on_batch_begin for all callbacks
                batch_logs = {}
                for callback in callbacks:
                    callback.on_batch_begin(batch, batch_logs)
                    
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Train on batch
                batch_loss, batch_metrics = self.train_on_batch(X_batch, y_batch)
                
                # Update batch logs
                batch_logs.update({
                    'loss': batch_loss,
                    **batch_metrics
                })
                
                # Call on_batch_end for all callbacks
                for callback in callbacks:
                    callback.on_batch_end(batch, batch_logs)
                    
                # Update epoch metrics
                batch_size_actual = end_idx - start_idx
                epoch_loss += batch_loss * batch_size_actual
                
                # Accumulate metrics
                for metric, value in batch_metrics.items():
                    if metric not in epoch_metrics:
                        epoch_metrics[metric] = 0
                    epoch_metrics[metric] += value * batch_size_actual
                
            # Calculate average metrics for the epoch
            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)
            
            for metric in epoch_metrics:
                epoch_metrics[metric] /= n_samples
                history[metric].append(epoch_metrics[metric])
            
            # Prepare logs
            logs = {
                'loss': epoch_loss,
                **epoch_metrics
            }
            
            # Evaluate on validation data if provided
            if validation_data:
                val_metrics = self.evaluate(*validation_data, batch_size=batch_size)
                
                for metric, value in val_metrics.items():
                    val_metric_name = f'val_{metric}'
                    logs[val_metric_name] = value
                    history[val_metric_name].append(value)
                
            # Print epoch results if verbose
            if verbose:
                epoch_str = f"Epoch {epoch+1}/{epochs} - "
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                print(f"{epoch_str}{metrics_str}")
                
            # Call on_epoch_end for all callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)
                
        # Call on_train_end for all callbacks
        for callback in callbacks:
            callback.on_train_end(logs)
            
        return history
        
    def predict(self, X, batch_size=32):
        """Make predictions on new data"""
        if not self._compiled:
            raise ValueError("Model must be compiled before making predictions.")
            
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        predictions = []
        
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            
            # Forward pass with training=False
            batch_predictions = self.forward(X_batch, training=False)
            predictions.append(batch_predictions)
            
        # Concatenate all batch predictions
        return np.vstack(predictions)
    
    def save(self, filepath):
        """Save model to a file"""
        if not self._compiled:
            raise ValueError("Model must be compiled before saving.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        # Create a model representation
        model_data = {
            'layers': [],
            'optimizer': {
                'class': self.optimizer.__class__.__name__,
                'config': self.optimizer.get_config()
            },
            'loss': self.loss_function.__class__.__name__
        }
        
        # Save each layer's configuration and weights
        for layer in self.layers:
            layer_data = {
                'name': layer.name,
                'class': layer.__class__.__name__,
                'config': layer.get_config()
            }
            
            if layer.trainable:
                layer_data['weights'] = [param.tolist() for param in layer.get_parameters()]
                
            model_data['layers'].append(layer_data)
                
        try:
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def save_weights(self, filepath):
        """Save only the weights to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get model weights
        weights = self.get_weights()
        
        try:
            # Save weights to file
            with open(filepath, 'wb') as f:
                pickle.dump(weights, f)
            return True
        except Exception as e:
            print(f"Error saving weights: {str(e)}")
            return False
    
    def load_weights(self, filepath):
        """Load weights from a file"""
        try:
            # Check if filepath is a pickle file or JSON
            if filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    model_data = json.load(f)
                
                # Extract weights from model data
                weights_data = []
                for layer_data in model_data['layers']:
                    if 'weights' in layer_data:
                        weights_data.append((layer_data['name'], 
                                           [np.array(w) for w in layer_data['weights']]))
            else:
                # Assume it's a pickle file with weights
                with open(filepath, 'rb') as f:
                    weights_data = pickle.load(f)
                    
            # Load weights
            return self.set_weights(weights_data)
            
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """Load a model from a file"""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
                
            # Create a new model
            model = cls()
            
            # Create and add layers
            for layer_data in model_data['layers']:
                # Create layer based on class name
                layer_class = globals()[layer_data['class']]
                
                # Extract layer config
                layer_config = layer_data['config']
                
                # Remove class from config
                if 'class' in layer_config:
                    del layer_config['class']
                    
                # Handle activation layers specially
                if layer_data['class'] == 'Activation':
                    activation_class = globals()[layer_config['activation']]
                    layer = layer_class(activation_class(), name=layer_config['name'])
                else:
                    # Create layer with config
                    layer = layer_class(**layer_config)
                
                # Add layer to model
                model.add(layer)
            
            # Create optimizer
            optimizer_data = model_data['optimizer']
            optimizer_class = globals()[optimizer_data['class']]
            optimizer = optimizer_class(**optimizer_data['config'])
            
            # Create loss function
            loss_class = globals()[model_data['loss']]
            loss = loss_class()
            
            # Compile model
            model.compile(loss=loss, optimizer=optimizer)
            
            # Load weights if present
            for i, layer_data in enumerate(model_data['layers']):
                if layer_data['class'] == 'Activation':
                    continue  # Skip activation layers as they don't have weights
                    
                if 'weights' in layer_data and model.layers[i].trainable:
                    for j, param in enumerate(model.layers[i].get_parameters()):
                        if j < len(layer_data['weights']):
                            weights = np.array(layer_data['weights'][j])
                            if param.shape == weights.shape:
                                param[:] = weights
                            else:
                                print(f"Warning: Shape mismatch for {layer_data['name']}, parameter {j}. " +
                                     f"Expected {param.shape}, got {weights.shape}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    
    def summary(self):
        """Print a summary of the model"""
        print("Model Summary:")
        print("=" * 80)
        print(f"{'Layer (type)':30} {'Output Shape':20} {'Param #':10}")
        print("=" * 80)
        
        total_params = 0
        trainable_params = 0
        input_shape = None
        
        for i, layer in enumerate(self.layers):
            # Get layer type
            layer_type = layer.__class__.__name__
            
            # Determine input shape for first layer
            if i == 0 and hasattr(layer, 'input_dim'):
                input_shape = (None, layer.input_dim)
            
            # Get output shape
            if hasattr(layer, 'output_dim'):
                output_shape = f"(None, {layer.output_dim})"
            elif input_shape is not None and layer_type == 'Activation':
                output_shape = str(input_shape)
            else:
                output_shape = "?"
                
            # Update input shape for next layer
            if hasattr(layer, 'output_dim'):
                input_shape = (None, layer.output_dim)
            
            # Get parameter count
            layer_params = 0
            for param in layer.get_parameters():
                layer_params += param.size
                
            total_params += layer_params
            if layer.trainable:
                trainable_params += layer_params
            
            print(f"{i+1}. {layer.name} ({layer_type}){'':<10} {output_shape:<20} {layer_params:<10}")
            
        print("=" * 80)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Non-trainable params: {total_params - trainable_params}")
        print("=" * 80)
        
    def plot_model(self, filename='model.png'):
        """Generate a plot of the model architecture"""
        try:
            import pydot
            
            # Create a new graph
            graph = pydot.Dot(graph_type='digraph', rankdir='TB')
            
            # Create nodes for each layer
            nodes = []
            for i, layer in enumerate(self.layers):
                layer_type = layer.__class__.__name__
                
                # Add additional info for specific layer types
                if hasattr(layer, 'input_dim') and hasattr(layer, 'output_dim'):
                    shape_info = f"\nInput: {layer.input_dim}, Output: {layer.output_dim}"
                else:
                    shape_info = ""
                    
                node_label = f"{layer.name}\n({layer_type}){shape_info}"
                
                # Create node
                node = pydot.Node(str(i), label=node_label, shape='box', 
                                 style='filled', fillcolor='lightblue' if layer.trainable else 'lightgrey')
                graph.add_node(node)
                nodes.append(node)
                
                # Connect to previous layer if not the first layer
                if i > 0:
                    edge = pydot.Edge(nodes[i-1], nodes[i])
                    graph.add_edge(edge)
                    
            # Save the graph
            graph.write_png(filename)
            print(f"Model plot saved to {filename}")
            
        except ImportError:
            print("Error: pydot and graphviz are required to plot the model")
            
    def get_weights(self):
        """Get all weights from the model"""
        weights = []
        for layer in self.layers:
            if layer.trainable:
                layer_weights = [param.copy() for param in layer.get_parameters()]
                weights.append((layer.name, layer_weights))
        return weights
    
    def set_weights(self, weights):
        """Set weights for the model"""
        success = True
        
        for layer_name, layer_weights in weights:
            # Find corresponding layer by name
            layer_found = False
            for layer in self.layers:
                if layer.name == layer_name and layer.trainable:
                    layer_found = True
                    # Set layer parameters
                    for i, param in enumerate(layer.get_parameters()):
                        if i < len(layer_weights):
                            # Check shapes match
                            if param.shape == layer_weights[i].shape:
                                param[:] = layer_weights[i]
                            else:
                                print(f"Warning: shape mismatch for {layer.name}, parameter {i}. " +
                                     f"Expected {param.shape}, got {layer_weights[i].shape}")
                                success = False
                    break
                    
            if not layer_found:
                print(f"Warning: layer {layer_name} not found in model")
                success = False
                
        return success


# Example usage: Binary classification on the moons dataset
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate a non-linear dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Reshape y to be (n_samples, 1)
    y = y.reshape(-1, 1)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create a directory for model checkpoints
    os.makedirs("model_checkpoints", exist_ok=True)
    
    # Create a model
    model = NeuralNetwork()
    
    # Add layers
    model.add(Dense(2, 32, name="input_layer"))
    model.add(Activation(ReLU(), name="relu1"))
    model.add(BatchNormalization(32, name="batchnorm1"))
    model.add(Dense(32, 16, name="hidden_layer1"))
    model.add(Activation(ReLU(), name="relu2"))
    model.add(Dropout(0.3, name="dropout1"))
    model.add(Dense(16, 8, name="hidden_layer2"))
    model.add(Activation(ReLU(), name="relu3"))
    model.add(Dense(8, 1, name="output_layer"))
    model.add(Activation(Sigmoid(), name="sigmoid_output"))
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001),
        ModelCheckpoint(
            filepath="model_checkpoints/model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.json",
            monitor='val_loss',
            save_best_only=True
        ),
        LearningRateScheduler(
            schedule=lambda epoch: 0.01 * (0.9 ** epoch),
            verbose=1
        ),
        TensorBoard(log_dir='logs/run1', histogram_freq=5),
        ProgressBar(total_epochs=100)
    ]
    
    # Compile the model
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.01)
    )
    
    try:
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0  # Disable default verbose output since we're using ProgressBar
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training history
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot precision if available
        if 'precision' in history and 'val_precision' in history:
            plt.subplot(2, 2, 3)
            plt.plot(history['precision'], label='Training Precision')
            plt.plot(history['val_precision'], label='Validation Precision')
            plt.title('Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
        
        # Plot recall if available
        if 'recall' in history and 'val_recall' in history:
            plt.subplot(2, 2, 4)
            plt.plot(history['recall'], label='Training Recall')
            plt.plot(history['val_recall'], label='Validation Recall')
            plt.title('Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True)
