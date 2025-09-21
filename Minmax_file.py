import numpy as np


class MinMax:
    def __init__(self, main, depth=0):
        self.main = main  # passerelle avec main
        self.depth = depth  # profondeur maximale de la recherche Minmax
        self.corner_distances = [[(-j, -i) if i < 4 and j < 4 else (-j, 7 - i) if j < 4 else (7 - j, -i) if i < 4 else (7 - j, 7 - i) for i in range(8)] for j in range(8)]

    def minmax(self, board, depth, alpha, beta, maximizing_player):
        board_save = board.copy() # sauvegarde du plateau

        mixed_board, return_board = self.main.game.check_valid_play(board, self.main.game.player_turn if maximizing_player else - self.main.game.player_turn)  # calcul des coups jouables

        # cases où mixed_board est égal à 2 (coups jouables)
        valid_moves = np.argwhere(mixed_board == 2)

        # vérification que la partie soit terminée ou que l'on ai atteint la profondeur max
        if not valid_moves.size:
            return self.evaluate(mixed_board, ended=True)  # évaluation de l'état du jeu
        elif depth == 0:
            return self.evaluate(mixed_board)  # évaluation de l'état du jeu

        if maximizing_player: # tour de l'agent Minmax
            max_eval = float('-inf')  # valeur maximale initiale

            for move in valid_moves:  # boucle sur les coups jouables
                board[move[0], move[1]] = self.main.game.player_turn  # application du coup
                board = self.main.game.do_return_pieces(move[0], move[1], board, return_board, self.main.game.player_turn, treesearch=True)
                eval = self.minmax(board, depth - 1, alpha, beta, False)  # évaluation de la position
                board = board_save.copy() # annulation du coup
                max_eval = max(max_eval, eval)  # mise à jour de la meilleure évaluation
                alpha = max(alpha, eval)  # mise à jour d'alpha
                if beta <= alpha:  # élagage
                    break

            return max_eval  # retourne la meilleure évaluation trouvée

        else: # tour de l'adversaire
            min_eval = float("inf")  # valeur minimale initiale

            for move in valid_moves:  # boucle sur les coups jouables
                board[move[0], move[1]] = -self.main.game.player_turn  # application du coup
                board = self.main.game.do_return_pieces(move[0], move[1], board, return_board, -self.main.game.player_turn, treesearch=True)
                eval = self.minmax(board, depth - 1, alpha, beta, True)  # évaluation de la position
                board = board_save.copy()  # annulation du coup
                min_eval = min(min_eval, eval)  # mise à jour de la pire évaluation
                beta = min(beta, eval)  # mise à jour de beta
                if beta <= alpha:  # élagage
                    break

            return min_eval  # retourne la pire évaluation trouvée

    # fonction déterminant le meilleur mouvement (selon l'agent)
    def find_best_move(self, board):
        best_move = None  # initialisation du meilleur coup
        best_value = float('-inf')  # initialisation de la meilleure évaluation
        board_save = board.copy() # sauvegarde du plateau

        # cases où mixed_board est égal à 2
        valid_moves = np.argwhere(self.main.game.mixed_board == 2)

        for move in valid_moves:  # boucle sur les coups jouables
            board[move[0], move[1]] = self.main.game.player_turn # application du coup
            board = self.main.game.do_return_pieces(move[0], move[1], board, self.main.game.return_board, self.main.game.player_turn, treesearch=True)
            move_value = self.minmax(board, self.depth, float('-inf'), float('inf'), False)  # évaluation du coup
            board = board_save.copy() # annulation du coup

            if move_value > best_value:  # mise à jour du meilleur coup si nécessaire
                best_value = move_value
                best_move = move
        return best_move  # retourne le meilleur coup trouvé

    # fonction d'évaluation d'une position
    def evaluate(self, mixed_board, ended=False):
        player_turn = self.main.game.player_turn

        if not ended:
            mixed_board[mixed_board == 2] = 0
            score = 0

            not_zero = np.argwhere((mixed_board != 0) & (mixed_board != 2))

            for x, y in not_zero:
                distance = self.corner_distances[x][y]
                new = 0

                # coins
                if abs(distance[0]) <= 0 and abs(distance[1]) <= 0:
                    new += 25

                # 1 de distances des coins, sur les bords
                elif abs(distance[0]) <= 1 and abs(distance[1]) <= 0 or abs(distance[0]) <= 0 and abs(distance[1]) <= 1:
                    new += 8 + (60 - np.sum(mixed_board == 0)) // (60 / 5) if mixed_board[x + distance[0], y + distance[1]] == mixed_board[x, y]\
                        else 6 + (60 - np.sum(mixed_board == 0)) // (60 / 3) if np.all(mixed_board[min(x, x - distance[0] * 6): max(x, x - distance[0] * 6) + 1, min(y, y - distance[1] * 6): max(y, y - distance[1] * 6) + 1] == mixed_board[x, y]) and \
                            mixed_board[x + distance[0], y + distance[1]] == 0 \
                        else 3 + (60 - np.sum(mixed_board == 0)) // (60 / 3) if np.all(mixed_board[min(x, x - distance[0] * 4): max(x, x - distance[0] * 4) + 1, min(y, y - distance[1] * 4): max(y, y - distance[1] * 4) + 1] == mixed_board[x, y]) and \
                            mixed_board[x - distance[0] * 6, y - distance[1] * 6] == 0 \
                        else -18

                # 1 de distance des coins, au milieu
                elif abs(distance[0]) <= 1 and abs(distance[1]) <= 1:
                    new += 6 + (60 - np.sum(mixed_board == 0)) // (60 / 5) if mixed_board[x + distance[0], y + distance[1]] == mixed_board[x, y] \
                        else -22

                # 2 de distance des coins, sur les bords
                elif abs(distance[0]) <= 2 and abs(distance[1]) <= 0 or abs(distance[0]) <= 0 and abs(distance[1]) <= 2:
                    new += 8 + (60 - np.sum(mixed_board == 0)) // (60 / 5) if (mixed_board[x + distance[0], y + distance[1]] == mixed_board[x, y]
                            and mixed_board[int(x + distance[0] * 0.5), int(y + distance[1] * 0.5)] == mixed_board[x, y]) \
                        else 7 + (60 - np.sum(mixed_board == 0)) // (60 / 4)

                # 2 de distance des coins, 1 de distance des bords
                elif abs(distance[0]) <= 2 and abs(distance[1]) <= 1 or abs(distance[0]) <= 1 and abs(distance[1]) <= 2:
                    new += 1.5 if (mixed_board[x + distance[0], y + distance[1]] == mixed_board[x, y]
                            and mixed_board[x + distance[0] if abs(distance[0]) == 1 else x, y + distance[1] if abs(distance[1]) == 1 else y] == mixed_board[x, y]) \
                        else 1 if mixed_board[x + distance[0], y + distance[1]] == mixed_board[x, y] \
                        else 2 if mixed_board[int(x + abs(int(distance[1] / 2)) * distance[0] / abs(distance[0])), int(y + abs(int(distance[0] / 2)) * distance[1] / abs(distance[1]))] == mixed_board[x, y] \
                        else -7 - (60 - np.sum(mixed_board == 0)) // (60 / 5)

                # 2 de distance des coins, au milieu
                elif abs(distance[0]) <= 2 and abs(distance[1]) <= 2:
                    new += 1 if mixed_board[x + distance[0], y + distance[1]] == mixed_board[x, y] \
                        else 4 - (60 - np.sum(mixed_board == 0)) // (60 / 4)

                # 3 de distance des coins, sur les bords
                elif abs(distance[0]) <= 3 and abs(distance[1]) <= 0 or abs(distance[0]) <= 0 and abs(distance[1]) <= 3:
                    new += 8 + (60 - np.sum(mixed_board == 0)) // (60 / 5) if np.all(mixed_board[min(x, x + distance[0]): max(x, x + distance[0]) + 1,min(y, y + distance[1]): max(y, y + distance[1]) + 1] == mixed_board[x, y]) \
                            or np.all(mixed_board[min(x, x - int(distance[0] * 4 / 3)): max(x, x - int(distance[0] * 4 / 3)) + 1, min(y, y - int(distance[1] * 4 / 3)): max(y, y - int(distance[1] * 4 / 3) + 1)] == mixed_board[x, y]) \
                        else 5 + (60 - np.sum(mixed_board == 0)) // (60 / 5) if np.all(mixed_board[min(x + distance[0] // 3, x - distance[0] // 3): max(x + distance[0] // 3, x - distance[0] // 3) + 1, min(y + distance[1] // 3,y - distance[1] // 3): max(y + distance[1] // 3, y - distance[1] // 3) + 1]) != -mixed_board[x, y] \
                        else 0

                # 3 de distance des coins, 1 de distance des bords
                elif abs(distance[0]) <= 3 and abs(distance[1]) <= 1 or abs(distance[0]) <= 1 and abs(distance[1]) <= 3:
                    new += 1 if mixed_board[int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0])), int(
                        y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]))] == mixed_board[x, y] \
                        else -1 - (60 - np.sum(mixed_board == 0)) // (60 / 3) if np.all(mixed_board[int(x + abs(int(distance[1] / 3)) *distance[0] / abs(distance[0]) + min(int(distance[0] / 3),-int(distance[0] / 3))): \
                            int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + max(int(distance[0] / 3), -int(distance[0] / 3)) + 1), \
                            int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + min(int(distance[1] / 3),-int(distance[1] / 3))): \
                            int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + max((distance[1] / 3), -int(distance[1] / 3)) + 1)] != -mixed_board[x, y]) \
                            and np.any(mixed_board[int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + min(int(distance[0] / 3),-int(distance[0] / 3))): \
                            int(x + abs(int(distance[1] / 3)) * distance[0] / abs(distance[0]) + max(int(distance[0] / 3), -int(distance[0] / 3)) + 1), \
                            int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + min(int(distance[1] / 3), -int(distance[1] / 3))): \
                            int(y + abs(int(distance[0] / 3)) * distance[1] / abs(distance[1]) + max(int(distance[1] / 3), -int(distance[1] / 3)) + 1)] == mixed_board[x, y]) \
                        else -6 - (60 - np.sum(mixed_board == 0)) // (60 / 4)

                # 3 de distance des coins, 2 de distance des bords
                elif abs(distance[0]) <= 3 and abs(distance[1]) <= 2 or abs(distance[0]) <= 2 and abs(distance[1]) <= 3:
                    new += 1

                # 3 de distance des coins, au milieu
                elif abs(distance[0]) <= 3 and abs(distance[1]) <= 3:
                    new += 1

                if mixed_board[x, y] == player_turn:
                    score += new
                else:
                    score -= new if mixed_board[x, y] == player_turn else max(-1, new)

        else:
            score = np.sum(mixed_board == player_turn) - np.sum(mixed_board == -player_turn)

        return score