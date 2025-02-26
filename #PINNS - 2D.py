#PINNS

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Définition du réseau de neurones
class PINN(nn.Module):
    def __init__(self, n_hidden=3, n_neurons=20):
        super(PINN, self).__init__()
        layers = []
        input_dim = 2  # (x, y)
        output_dim = 1  # u(x, y)
        
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Sigmoid())
        
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Sigmoid())
        
        layers.append(nn.Linear(n_neurons, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Définition de la loss fonction pour les conditions aux limites
def boundary_loss(model, boundary_points, boundary_values):
    u_pred = model(boundary_points)
    return torch.mean((u_pred - boundary_values) ** 2)

# Définition de la loss physique basée sur l'équation de Laplace
def physical_loss(model, interior_points):
    interior_points.requires_grad = True
    u_pred = model(interior_points)
    
    grads = torch.autograd.grad(u_pred, interior_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(grads[:, 0], interior_points, grad_outputs=torch.ones_like(grads[:, 0]), create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(grads[:, 1], interior_points, grad_outputs=torch.ones_like(grads[:, 1]), create_graph=True)[0][:, 1]
    
    return torch.mean((u_xx + u_yy) ** 2)

# Entraînement du modèle
def train(model, optimizer, boundary_points, boundary_values, interior_points, epochs=10000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_b = boundary_loss(model, boundary_points, boundary_values)
        loss_p = physical_loss(model, interior_points)
        loss = loss_b + loss_p
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, Boundary Loss = {loss_b.item():.6f}, Physical Loss = {loss_p.item():.6f}")

# Génération des données de test
boundary_points = torch.tensor([[0, y] for y in torch.linspace(0, 1, 10)] + [[1, y] for y in torch.linspace(0, 1, 10)], dtype=torch.float32)
boundary_values = torch.tensor([1.0] * 10 + [0.0] * 10, dtype=torch.float32).unsqueeze(1)
interior_points = torch.rand((100, 2)) #100 pour le nombre de points intérieurs et 2 pour les coordonnées x et y
print("boundarypoints", boundary_points)
print("boundaryvalues", boundary_values)
print("interiorpoints", interior_points)

# Initialisation et entraînement du modèle
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train(model, optimizer, boundary_points, boundary_values, interior_points)

# Visualisation des résultats
def plot_solution(model):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    points_tensor = torch.tensor(points, dtype=torch.float32)
    U_pred = model(points_tensor).detach().numpy().reshape(50, 50)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, U_pred, levels=50, cmap='viridis')
    plt.colorbar(label='Potential u(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution de léquation de Laplace')
    plt.show()

# Affichage de la solution
plot_solution(model)