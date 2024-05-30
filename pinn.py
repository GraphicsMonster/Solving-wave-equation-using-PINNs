# Solving the wave equation defined in 1 spatial dimension.
# Here are the boundary and initial conditions
'''
1. u(0, t) = 0
2. u(1, t) = 0
3. u(x, 0) = sin(pi*x)
4. du(x, 0)/dt = 0
'''

import torch
import torch.nn as nn

# Hyperparams
velocity = 1
num_epochs = 1000


def analytical_solution(x, t):
    return torch.sin(torch.pi * x) * torch.cos(torch.pi * t)


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PINN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, t):
        x = torch.concat([x, t], dim=1)
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        out = self.tanh(out)
        out = self.linear4(out)
        return out
    
def pde_residual(model, x, t, velocity):
    u = model(x, t)
    x.requires_grad = True
    t.requires_grad = True
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

    residual = u_tt - (velocity**2)*u_xx
    loss = torch.mean(torch.square(residual))
    return loss

def boundary_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    # ideally all u's here would be 0 considering the training on this part happens such that half the x values are 0 and the other half are 1.
    loss = torch.mean(torch.square(u))
    return loss

def initial_loss(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    correct_val = torch.sin(torch.pi * x)
    loss_1 = torch.mean(torch.square(correct_val - u))

    loss_2 = torch.mean(torch.square(torch.autograd.grad(u, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]))

    total_loss = loss_1 + loss_2
    return total_loss

# Generating data for pde training
x_colloc = torch.rand(1000, 1, requires_grad=True)
t_colloc = torch.rand(1000, 1, requires_grad=True)

# Generating data for bc_training
x_bc = torch.cat([torch.zeros(500, 1), torch.ones(500, 1)], dim=0)
t_bc = torch.rand(1000, 1, requires_grad=True)

# Generating data for initial_conditions training
x_ic = torch.rand(1000, 1, requires_grad=True)
t_ic = torch.zeros(1000, 1, requires_grad=True)

model = PINN(2, 25)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()

    pde_loss = pde_residual(model, x_colloc, t_colloc, velocity)
    bc_loss = boundary_loss(model, x_bc, t_bc)
    init_loss = initial_loss(model, x_ic, t_ic)
    total_loss = pde_loss + bc_loss + init_loss

    total_loss.backward()
    optimizer.step()

    if epoch%100 == 0:
        print(f"epoch:{epoch}/{num_epochs} || loss:{total_loss.item()}")

# Generating data to compare the analytical solution and the model predictions
import matplotlib.pyplot as plt

x = torch.linspace(0, 1, 1000)
t = torch.linspace(0, 1, 1000)

X, T = torch.meshgrid(x, t, indexing='ij')

X_flat = X.flatten().view(-1, 1)
T_flat = T.flatten().view(-1, 1)

u_analytic = analytical_solution(X_flat, T_flat)
u_pred = model(X_flat, T_flat).detach()

u_analytic = u_analytic.view(1000, 1000)
u_pred = u_pred.view(1000, 1000)

# Plotting the curves
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X.numpy(), T.numpy(), u_analytic.numpy(), cmap='viridis')
ax.set_title("Analytical Solution")
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X.numpy(), T.numpy(), u_pred.numpy(), cmap='viridis')
ax.set_title("Model Predictions")
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')

plt.show()


# The model predictions are pretty close to the analytical solution.
# Let us plot the evolution of the displacement with varying spatial coordinates as time evolves in the background with no axis labels. Kind of like an animation

import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111)

line1, = ax.plot(x.numpy(), u_analytic[:, 0].numpy(), 'r', label='Analytical')
line2, = ax.plot(x.numpy(), u_pred[:, 0].numpy(), 'b', label='Predictions')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_ylim(-1, 1)
ax.legend()

def update(i):
    line1.set_ydata(u_analytic[:, i].numpy())
    line2.set_ydata(u_pred[:, i].numpy())
    ax.set_title(f"Time: {t[i].item()}")
    return line1, line2

ani = animation.FuncAnimation(fig, update, frames=range(1000), interval=50)
plt.show()