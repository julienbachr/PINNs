import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Définition du réseau de neurones pour le problème 1D
class PINN_1D(nn.Module):
    def __init__(self, n_hidden=3, n_neurons=20): #3 couches et 20 neurones par couche, suffisant pour ce cas simple
        super(PINN_1D, self).__init__()
        layers = [nn.Linear(1, n_neurons), nn.Sigmoid()]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid()) #fonction d'activation
        layers.append(nn.Linear(n_neurons, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Définition des fonctions de perte
def boundary_loss(model, boundary_points, boundary_values):
    u_pred = model(boundary_points)
    return torch.mean((u_pred - boundary_values) ** 2)

def physical_loss(model, interior_points, rho, epsilon):
    interior_points.requires_grad = True
    u_pred = model(interior_points)
    u_x = torch.autograd.grad(u_pred, interior_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, interior_points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return torch.mean((u_xx + rho(interior_points) / epsilon) ** 2)

# Profils de charge
rho_0 = 1.0  # Densité de charge
epsilon = 1.0  # Permittivité

# Charge uniforme
def rho_uniform(x):
    return torch.full_like(x, rho_0)

# Entraînement du modèle
def train(model, optimizer, boundary_points, boundary_values, interior_points, rho, epochs=1000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_b = boundary_loss(model, boundary_points, boundary_values)
        loss_p = physical_loss(model, interior_points, rho, epsilon)
        loss = loss_b + loss_p
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, Boundary Loss = {loss_b.item():.6f}, Physical Loss = {loss_p.item():.6f}")

# Génération des données de test
boundary_points = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
boundary_values = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
interior_points = torch.rand((100, 1))

# Initialisation et entraînement du modèle pour une charge uniforme
model = PINN_1D()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train(model, optimizer, boundary_points, boundary_values, interior_points, rho_uniform)

# Visualisation des résultats
def plot_solution(model, label):
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    u_pred = model(x_tensor).detach().numpy()
    u_exact = (rho_0 / (2 * epsilon)) * (x - x ** 2)  # Solution analytique pour rho uniforme
    
    plt.plot(x, u_pred, label='PINN Solution', linestyle='dashed', zorder=2)
    plt.plot(x, u_exact, label='Exact Solution', linestyle='solid', zorder=1)
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.legend()
    plt.title(f'Solution de l\'équation de Poisson en 1D ({label})')
    plt.show()

# Affichage de la solution avec charge uniforme
plot_solution(model, "Charge uniforme")