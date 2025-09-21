import torch
from torch import nn
from torch.optim import Adam
from collections import deque
import pickle
import os
import random
import numpy as np
from math import exp


# Classe servant de buffer (mémoire temporaire)
class ReplayMemory:
    def __init__(self, ddqn, capacity):
        self.ddqn = ddqn # passerelle avec le modèle ddqn
        self.filename = self.ddqn.main.get_path("export/dqn_memory_file.pkl") # path de stockage du buffer
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # priorités des transitions
        self.position = 0  # index pour remplacer les transitions lorsque la mémoire est pleine
        self.capacity = capacity
        self.memory = deque(maxlen=capacity) # création du buffer
        self.load() # chargement du buffer existant

    # ajoute une transition au buffer
    def push(self, transition):
        # définir une priorité initiale élevée pour les nouvelles transitions
        max_priority = self.priorities.max() if self.memory else 1.0
        max_priority = min(max(max_priority, 1e-6), 1e3)

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    # renvoie une batch depuis le buffer, plus la priorité est élevée, plus la transition a de chance d'être choisit
    def sample(self, batch_size):
        # calcul d'alpha qui diminue avec le temps
        alpha = max(0.6 - (0.0000007 * self.ddqn.total_steps), 0.4)  # réduit alpha avec le temps, minimum à 0.4

        if len(self.memory) == self.capacity:
            probabilities = self.priorities ** alpha  # alpha pour ajuster le biais
        else:
            probabilities = np.maximum(self.priorities[:len(self.memory)], 1e-6) ** alpha

        sum_proba = probabilities.sum()
        if sum_proba > 0:
            probabilities /= sum_proba
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)  # probabilités uniformes

        # mélange des probabilités basées sur les priorités et uniformes
        uniform_probabilities = np.ones_like(probabilities) / len(probabilities)
        probabilities = 0.9 * probabilities + 0.1 * uniform_probabilities  # mix des proba

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[idx] for idx in indices]

        # importance-sampling weights
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** -0.4  # beta pour ajuster la correction
        weights /= weights.max()  # normalise pour éviter de grandes valeurs
        return transitions, indices, weights

    def update_priorities(self, indices, priorities):
        # Clamp les TD errors pour éviter des valeurs extrêmes
        clipped_priorities = np.clip(priorities.cpu().detach().numpy(), a_min=1e-6, a_max=1e3)  # Limites raisonnables
        for idx, priority in zip(indices, clipped_priorities):
            self.priorities[idx] = priority

    # permet de renvoyer len(self.memory) pour len(ReplayMemory)
    def __len__(self):
        return len(self.memory)

    # sauvegarde le buffer dans un fichier externe
    def save(self):
        try:
            with open(self.filename, "wb") as f:
                pickle.dump((list(self.memory), self.priorities, self.position), f)
                print(f"Mémoire sauvegardée à {self.filename}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")

    # charge le buffer d'un fichier externe
    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "rb") as f:
                    mem_list, self.priorities, self.position = pickle.load(f)
                    self.memory = deque(mem_list, maxlen=self.capacity)
                    print(f"Mémoire chargée depuis {self.filename}")
            except Exception as e:
                print(f"Erreur lors du chargement : {e}")
        else:
            print(f"Fichier {self.filename} non trouvé. Initialisation d'une nouvelle mémoire.")


# agent de Double Deep Q-Network (deep reinforcement learning) avec Prioritized Replay Memory et Cycling Epsilon-greedy
class DDQN(nn.Module):
    def __init__(self, main, double=False):
        super(DDQN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.main = main # passerelle avec main
        self.logging_scores = [] # suivit des scores du modèle DQN
        self.logging_rewards = [] # suivit des récompenses modèle DQN
        self.batch_size = 64 # taille des batchs
        self.done = False # vérifie si la partie est finie au prochain tour

        self.main_network = DQN(device=self.device).to(self.device)
        self.target_network = DQN(device=self.device).to(self.device) if double else None

        self.memory = ReplayMemory(self, 20000) # initialisation du buffer
        self.temp_memory = []
        self.total_steps = 0 # suit le nombre de pas d'apprentissage

        self.epsilon_min = 0.05 # valeur minimal d'epsilon
        self.epsilon_max = 1.00
        self.epsilon = self.epsilon_max  # facteur epsilon influant la variation entre exploration et exploitation
        self.target_update = 1000

        self.episode = 0  # indique la version de DQN
        if double:
            self.filename = self.main.get_path("export/dqn_model_file.pth") # path de stockage du modèle
        else:
            self.filename = self.main.get_path("export/dqn_model_file.pth") # path de stockage du modèle
        self.load_model()  # chargement de la dernière version du modèle
        self.update_target_network() if double else None # copie du main network vers le target_network

        self.corner_distances = [[(-j, -i) if i < 4 and j < 4 else (-j, 7 - i) if j < 4 else (7 - j, -i) if i < 4 else (7 - j, 7 - i) for i in range(8)] for j in range(8)]

    def get_epsilon(self):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * exp(-1. * self.episode / 3000)

    # fonction d'évaluation du plateau
    def evaluate_board(self, delta, board):
        score = 0
        not_zero = np.argwhere(delta != 0)

        for x, y in not_zero:
            distance = self.corner_distances[x][y]
            new = 0

            # coins
            if abs(distance[0]) <= 0 and abs(distance[1]) <= 0:
                new += 25

            # 1 de distances des coins, sur les bords
            elif abs(distance[0]) <= 1 and abs(distance[1]) <= 0 or abs(distance[0]) <= 0 and abs(distance[1]) <= 1:
                new += 8 + (60 - np.sum(board == 0)) // (60 / 5) if board[x + distance[0], y + distance[1]] == self.main.game.player_turn \
                    else 6 + (60 - np.sum(board == 0)) // (60 / 3)  if np.all(board[min(x, x - distance[0] * 6): max(x, x - distance[0] * 6) + 1, min(y, y - distance[1] * 6): max(y, y - distance[1] * 6) + 1] == self.main.game.player_turn) and  board[x + distance[0], y + distance[1]] == 0\
                    else 3 + (60 - np.sum(board == 0)) // (60 / 3) if np.all(board[min(x, x - distance[0] * 4): max(x, x - distance[0] * 4) + 1, min(y, y - distance[1] * 4): max(y, y - distance[1] * 4) + 1] == self.main.game.player_turn) and board[x - distance[0] * 6, y - distance[1] * 6] == 0 \
                    else -18

            # 1 de distance des coins, au milieu
            elif abs(distance[0]) <= 1 and abs(distance[1]) <= 1:
                new += 6 + (60 - np.sum(board == 0)) // (60 / 5) if board[x + distance[0], y + distance[1]] == self.main.game.player_turn \
                    else -22

            # 2 de distance des coins, sur les bords
            elif abs(distance[0]) <= 2 and abs(distance[1]) <= 0 or abs(distance[0]) <= 0 and abs(distance[1]) <= 2:
                new += 8 + (60 - np.sum(board == 0)) // (60 / 5) if (board[x + distance[0], y + distance[1]] == self.main.game.player_turn
                    and board[int(x + distance[0] * 0.5), int(y + distance[1] * 0.5)] == self.main.game.player_turn) \
                    else 7 + (60 - np.sum(board == 0)) // (60 / 4)

            # 2 de distance des coins, 1 de distance des bords
            elif abs(distance[0]) <= 2 and abs(distance[1]) <= 1 or abs(distance[0]) <= 1 and abs(distance[1]) <= 2:
                new += 1.5 if (board[x + distance[0], y + distance[1]] == self.main.game.player_turn
                    and board[x + distance[0] if abs(distance[0]) == 1 else x, y + distance[1] if abs(distance[1]) == 1 else y] == self.main.game.player_turn) \
                    else 1 - (60 - np.sum(board == 0)) // (60 / 3) if board[x + distance[0], y + distance[1]] == self.main.game.player_turn \
                    else 2 if board[int(x + abs(int(distance[1] / 2)) * distance[0] / abs(distance[0])), int(y + abs(int(distance[0] / 2)) * distance[1] / abs(distance[1]))] == self.main.game.player_turn \
                    else -7 - (60 - np.sum(board == 0)) // (60 / 5)

            # 2 de distance des coins, au milieu
            elif abs(distance[0]) <= 2 and abs(distance[1]) <= 2:
                new += 1 - (60 - np.sum(board == 0)) // (60 / 2) if board[x + distance[0], y + distance[1]] == self.main.game.player_turn \
                    else 4 - (60 - np.sum(board == 0)) // (60 / 4)

            # 3 de distance des coins, sur les bords
            elif abs(distance[0]) <= 3 and abs(distance[1]) <= 0 or abs(distance[0]) <= 0 and abs(distance[1]) <= 3:
                new += 8 + (60 - np.sum(board == 0)) // (60 / 5) if np.all(board[min(x, x + distance[0]): max(x, x + distance[0]) + 1, min(y, y + distance[1]): max(y, y + distance[1]) + 1] == self.main.game.player_turn) \
                    or np.all(board[min(x, x - int(distance[0] * 4 / 3)): max(x, x - int(distance[0] * 4 / 3)) + 1, min(y, y - int(distance[1] * 4 / 3)): max(y, y - int(distance[1] * 4 / 3) + 1)] == self.main.game.player_turn) \
                    else 5 + (60 - np.sum(board == 0)) // (60 / 5) if np.all(board[min(x + distance[0] // 3, x - distance[0] // 3): max(x + distance[0] // 3, x - distance[0] // 3) + 1, min(y + distance[1] // 3, y - distance[1] // 3): max(y + distance[1] // 3, y - distance[1] // 3) + 1]) != -self.main.game.player_turn \
                    else 0

            # 3 de distance des coins, 1 de distance des bords
            elif abs(distance[0]) <= 3 and abs(distance[1]) <= 1 or abs(distance[0]) <= 1 and abs(distance[1]) <= 3:
                new += 1 if  board[int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0])), int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]))] == self.main.game.player_turn \
                    else -1 - (60 - np.sum(board == 0)) // (60 / 3) if np.all(board[int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + min(int(distance[0] / 3), -int(distance[0] / 3))): \
                        int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + max(int(distance[0] / 3), -int(distance[0] / 3)) + 1), \
                        int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + min(int(distance[1] / 3), -int(distance[1] / 3))): \
                        int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + max(int(distance[1] / 3), -int(distance[1] / 3)) + 1)] != -self.main.game.player_turn)\
                    and np.any(board[int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + min(int(distance[0] / 3), -int(distance[0] / 3))): \
                        int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + max(int(distance[0] / 3), -int(distance[0] / 3)) + 1), \
                        int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + min(int(distance[1] / 3), -int(distance[1] / 3))): \
                        int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + max(int(distance[1] / 3), -int(distance[1] / 3)) + 1)] == self.main.game.player_turn) \
                    else -6 - (60 - np.sum(board == 0)) // (60 / 4)

            # 3 de distance des coins, 2 de distance des bords
            elif abs(distance[0]) <= 3 and abs(distance[1]) <= 2 or abs(distance[0]) <= 2 and abs(distance[1]) <= 3:
                new += 2 - (60 - np.sum(board == 0)) // (60 / 2)

            # 3 de distance des coins, au milieu
            elif abs(distance[0]) <= 3 and abs(distance[1]) <= 3:
                new += 1

            score += new

        return int(score)

    def get_reward(self, old_board, new_board, self_mixed_board, opponent_mixed_board):
        # plutot que de calculer l'entiereté de la valeur des positions, on s'intéresse uniquement aux cases retournés
        delta = new_board - old_board
        delta[(delta != 1) & (delta != -1)] //= 2

        gain = self.evaluate_board(delta, new_board)

        self_mobilities = np.sum(self_mixed_board == 2) if np.sum(self_mixed_board == 2) != 0 else -40
        opponent_mobilites = np.sum(opponent_mixed_board == 2) if np.sum(opponent_mixed_board == 2) != 0 else -40

        reward = (gain + min(max(self_mobilities - opponent_mobilites, -10), 10)) * 0.8 / 50

        return round(np.clip(reward, -0.8, 0.8), 2) # normalisation des récompenses pour éviter les gradients explosifs

    # fonction se chargeant d'améliorer l'agent
    def learn(self):
        # annule la fonction si l'agent n'a pas assez d'expérience
        if len(self.memory) < self.batch_size:
            return

        # prélèvement d'échantillons avec priorités
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        states, actions, rewards, next_states, dones, player_turns, mixed_states, mixed_next_states = batch # extraction des éléments du batch

        # transformation des éléments en tenseur
        states = torch.FloatTensor(np.array(states)).to(self.device) # tableau numpy
        actions = torch.LongTensor([x * 8 + y for x, y in actions]).unsqueeze(1).to(self.device) # un seul indice auquel on ajoute une dimension
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device) # tableau numpy
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        player_turns = torch.LongTensor(player_turns).to(self.device)
        mixed_states = torch.tensor(np.array(mixed_states), dtype=torch.float32).to(self.device)
        mixed_next_states = torch.tensor(np.array(mixed_next_states), dtype=torch.float32).to(self.device)

        q_values = self.main_network(states, player_turns, mixed_states).gather(1, actions) # sélectionne les Q-values associées aux actions effectuées, en se basant sur les indices des actions
        next_q_values = self.target_network(next_states, -player_turns, mixed_next_states).max(1)[0] # prédiction des Q-values puis détermination de la Q-value maximale pour chaque next_state
        target_q_values = rewards + (0.99 * next_q_values * (1 - dones)) # calcul de la Q-value cible

        # erreur TD et pondération
        td_errors = target_q_values.unsqueeze(1) - q_values
        loss = (weights * nn.SmoothL1Loss(reduction="none")(q_values, target_q_values.unsqueeze(1))).mean()

        # Backpropagation
        self.main_network.optimizer.zero_grad() # met à zéro les gradients de l'optimiseur
        loss.backward() # effectue la rétropropagation pour calculer les gradients de la perte
        for param in self.main_network.parameters():
            if param.grad is not None:
                torch.clamp(param.grad, -1.0, 1.0, out=param.grad)  # Limiter les gradients entre -1 et 1
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.0)  # Limiter les gradients
        self.main_network.optimizer.step() # mise à jour des paramètres du modèle en fonction des gradients calculés

        # mise à jour des priorités
        self.total_steps += 1
        new_priorities = torch.abs(td_errors.detach()) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        if self.episode % self.target_update == 0:
            self.update_target_network()

    # fonction sauvegardant le modèle
    def save_model(self):
        torch.save({ # sauvegarde des éléments suivants :
            "episode": self.episode, # l'épisode de l'agent
            "total_steps": self.total_steps, # le nombre de pas d'apprentissage
            "epsilon": self.epsilon, # la valeur d'epsilon
            "logging_rewards": self.logging_rewards, # les récompenses
            "model_state_dict": self.main_network.state_dict(), # les poids et les biais des neuronnes du résau sous la forme d'un dictionnaire
            "optimizer_state_dict": self.main_network.optimizer.state_dict(), # les paramètres de l'optimiseur
        }, self.filename)

        print("Modèle sauvegardé sous", self.filename)

    # fonction chargeant le modèle
    def load_model(self):
        checkpoint = torch.load(self.filename, weights_only=False) # chargement de tous les éléments du modèle
        self.main_network.load_state_dict(checkpoint["model_state_dict"]) # remplace les poids et biais actuelles par ceux récupérés
        self.main_network.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # recharge les paramètres de l'optimiseur
        self.episode = checkpoint["episode"] # récupère l'episode de l'agent
        self.total_steps = checkpoint["total_steps"] # récupère le nombre de pas d'apprentissage
        self.epsilon = checkpoint["epsilon"] # récupère la valeur d'epsilon
        self.logging_rewards = checkpoint["logging_rewards"] # récup_re les récompenses
        print("Modèle chargé depuis", self.filename)

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())


# résau neuronnal
class DQN(nn.Module):
    def __init__(self, device='cpu'):
        super(DQN, self).__init__()
        self.device = device

        # créations de 3 couches de neuronnes convolutifs traitant des matrices en 3 * 3 pour mieux capturer les motifs
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1) # reçois deux canneaux, d'une part le plateau, d'autre part la couleur de l'agent
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # sortie des couches convolutionnelles

        # créations de 3 couches de neuronnes fully connected traitant les données de manière linéaire
        self.fc1 = nn.Linear(64 * 8 * 8, 128) # reçois la sortie des couches convolutionnelles (64) et le concatenne au plateau (8*8)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64) # sortie des q-values estimées de chaque case du plateau

        self.optimizer = Adam(self.parameters(), lr=0.001) # utlisation de Adam comme optimiseur avec un taux d'apprentissage de 0.001

    def forward(self, x, player_turns, mixed_boards):
        # reshape de x pour s'assurer qu'il est de forme [N, 1, 8, 8] où:
        # N est le nombre d'échantillons, 1 est le canal unique (plateau), 8*8 est la taille du plateau
        x = x.view(-1, 1, 8, 8)

        # ajuste la forme de player_turns pour que chaque player_turn corresponde à chaque état
        player_turns = player_turns.view(-1, 1, 1, 1)  # Reshape pour [batch_size, 1, 1, 1]
        player_turns = player_turns.expand(-1, 1, 8, 8)  # Étendre pour [batch_size, 1, 8, 8]

        # concaténe x et player_turns
        x = torch.cat((x, player_turns), dim=1)  # x devient [batch_size, 2, 8, 8]

        # passage de 'x' à travers les couches convolutionnelles avec la fonction de sortie relu()
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # applatissement de 'x' pour préparer son entrée dans les couches fully connected
        # La nouvelle forme est [N, 64 * 8 * 8], où 64 est le nombre de filtres de la dernière couche conv
        x = x.view(-1, 64 * 8 * 8)

        # passage de 'x' à travers les couches fully connected avec la fonction de sortie relu()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Sortie finale

        if not isinstance(mixed_boards, torch.Tensor):
            mixed_boards = torch.tensor(mixed_boards, dtype=torch.float32, device=self.device)

        mask = (mixed_boards == 2).bool()
        mask = mask.view(-1, 64)

        x = x.masked_fill(~mask, -1e9)

        x = torch.clip(x, -1e3, 1e3)

        # retourne le plateau avec les q-values prédites de chaque case
        return x

    # fonction permettant à l'agent de choisir un coup
    def select_action(self, mixed_board, board, player_turn, epsilon):
        # liste des actions possibles
        available_actions = np.argwhere(mixed_board == 2)

        # si un nombre aléatoire est inférieur à epsilon, on choisit l'exploration
        if random.random() < epsilon:
            return random.choice(available_actions.tolist())

        # sinon, on choisit l'exploitation :

        # conversion en tenseurs
        board = torch.tensor(board, dtype=torch.float32, device=self.device)
        player_turn = torch.tensor([player_turn], dtype=torch.long, device=self.device)
        mixed_board = torch.tensor(mixed_board, dtype=torch.float32, device=self.device)

        # calcul des valeurs Q pour le plateau entier après avoir aplati prediction_board
        q_values = self(board.view(-1, 8 * 8).float(), player_turn, mixed_board).view(8, 8)

        # choisit l'action valide avec la meilleure valeur Q parmi available_actions
        best_action = max(available_actions, key=lambda a: q_values[a[0], a[1]].item())

        # retourne le coup optimal trouvé
        return best_action
