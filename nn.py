import numpy as np

class Tensor(object):
    def __init__(self, 
                 data, 
                 autograd=False, 
                 creators=None, 
                 creation_op=None, 
                 id=None):
        self.data = np.array(data)
        self.autograd = autograd
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None

        self.children = {}
        if id is None:
            id = np.random.randint(0, 100000)
        self.id = id

        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, count in self.children.items():
            if count != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("Cannont backpropagate more than once")
                else:
                    self.children[grad_origin.id] -= 1
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            assert grad.autograd is False

            if self.creators is not None and (
                self.all_children_grads_accounted_for() or grad_origin is None
            ):
                self.grad = grad
                if self.creation_op == "add":
                    self.creators[0].backward(grad=self.grad, grad_origin=self)
                    self.creators[1].backward(grad=self.grad, grad_origin=self)

                if self.creation_op == "neg":
                    self.creators[0].backward(
                        grad=self.grad.__neg__(), grad_origin=self
                    )

                if self.creation_op == "sub":
                    self.creators[0].backward(grad=self.grad, grad_origin=self)
                    self.creators[1].backward(
                        grad=self.grad.__neg__(), grad_origin=self
                    )

                if self.creation_op == "mul":
                    self.creators[0].backward(
                        grad=self.grad * self.creators[1], grad_origin=self
                    )
                    self.creators[1].backward(
                        grad=self.grad * self.creators[0], grad_origin=self
                    )

                if self.creation_op == "mm":
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(grad=new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, ds))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * 
                                              (self * (ones - self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if self.creation_op == "relu":
                    self.creators[0].backward(self.grad * (self > 0))

                if "HardTanh" in self.creation_op:
                    min_val, max_val = self.creation_op.split("_")[1:]
                    self.creators[0].backward(self.grad * (min_val <= self < max_val))

                if self.creation_op == "index_select":
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if self.creation_op == "cross_entropy":
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                data=self.data + other.data,
                autograd=True,
                creators=[self, other],
                creation_op="add",
            )
        else:
            return Tensor(data=self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(
                data=self.data * -1, autograd=True, creators=[self], creation_op="neg"
            )
        else:
            return Tensor(data=self.data * -1)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                data=self.data - other.data,
                autograd=True,
                creators=[self, other],
                creation_op="sub",
            )
        else:
            return Tensor(data=self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                data=self.data * other.data,
                autograd=True,
                creators=[self, other],
                creation_op="mul",
            )
        else:
            return Tensor(data=self.data * other.data)

    def sum(self, dim: int):
        if self.autograd:
            return Tensor(
                data=self.data.sum(dim),
                autograd=True,
                creators=[self],
                creation_op=f"sum_{dim}",
            )
        else:
            return Tensor(data=self.data.sum(dim))

    def expand(self, dim: int, copies: int):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.autograd:
            return Tensor(
                data=new_data,
                autograd=True,
                creators=[self],
                creation_op=f"expand_{dim}",
            )
        else:
            return Tensor(data=new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(
                data=self.data.transpose(),
                autograd=True,
                creators=[self],
                creation_op="transpose",
            )
        else:
            return Tensor(data=self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(
                data=self.data.dot(x.data),
                autograd=True,
                creators=[self, x],
                creation_op="mm",
            )
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if self.autograd:
            return Tensor(
                data=1 / (1 + np.exp(-self.data)),
                autograd=True,
                creators=[self],
                creation_op="sigmoid",
            )
        else:
            return Tensor(data=1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(
                data=np.tanh(self.data),
                autograd=True,
                creators=[self],
                creation_op="tanh",
            )
        else:
            return Tensor(data=np.tanh(self.data))

    def relu(self):
        if self.autograd:
            return Tensor(
                data=self.data * (self.data > 0),
                autograd=True,
                creators=[self],
                creation_op="relu",
            )
        else:
            return Tensor(data=self.data * (self.data > 0))

    def HardTanh(self, min_val: float = -1, max_val: float = 1):
        data = (
            (self.data > max_val)
            - (self.data < min_val)
            + self.data * (min_val <= self.data <= max_val)
        )
        if self.autograd:
            return Tensor(
                data=data,
                autograd=True,
                creators=[self],
                creation_op=f"HardTanh_{min_val}_{max_val}",
            )
        else:
            return Tensor(data=data)

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(
                self.data[indices.data],
                autograd=True,
                creators=[self],
                creation_op="index_select",
            )
            new.index_select_indices = indices
            return new
        else:
            return Tensor(self.data[indices.data])

    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(
            temp, axis=len(self.data.shape) - 1, keepdims=True
        )
        return softmax_output

    def cross_entropy(self, target_indices):

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(
            temp, axis=len(self.data.shape) - 1, keepdims=True
        )

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if self.autograd:
            out = Tensor(
                loss, autograd=True, creators=[self], creation_op="cross_entropy"
            )
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2 / n_inputs)
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.add.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for layer in self.layers:
            params += layer.get_parameters()
        return params


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()


class relu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.relu()


class HardTanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.HardTanh()


class Embedding(Layer):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)
        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)


class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)


class RNNCell(Layer):
    def __init__(
        self, n_inputs: int, n_hidden: int, n_output: int, activation="sigmoid"
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)

class LSTMCell(Layer):
    def __init__(self, n_inputs: int, n_hidden: int, n_output: int):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.xf = Linear(n_inputs, n_hidden)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)
        self.hf = Linear(n_hidden, n_hidden)
        self.hi = Linear(n_hidden, n_hidden)
        self.ho = Linear(n_hidden, n_hidden)
        self.hc = Linear(n_hidden, n_hidden)

        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()
        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hc.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        prev_hidden = hidden[0]
        prev_cell = hidden[1]

        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()
        c = (f * prev_cell) + (i * g)
        h = o * c.tanh()
        output = self.w_ho.forward(h)
        return output, (h, c)

    def init_hidden(self, batch_size=1):
        h = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        c = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        h.data[:,0] += 1
        c.data[:,0] += 1
        return (h, c)

class SGD(object):
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for parameter in self.parameters:
            parameter.grad.data *= 0

    def step(self, zero=True):
        for parameter in self.parameters:
            parameter.data -= parameter.grad.data * self.alpha
            if zero:
                parameter.grad.data *= 0
