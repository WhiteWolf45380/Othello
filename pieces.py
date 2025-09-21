import pygame


class Piece(pygame.sprite.Sprite):
    def __init__(self, game, number, x, y):
        super().__init__()
        self.game = game
        self.number = number
        self.color = "black" if self.number == -1 else "white"
        self.images = {
            -1: pygame.image.load(self.game.main.get_path(f"assets/black_piece.xcf")),
            1: pygame.image.load(self.game.main.get_path(f"assets/white_piece.xcf")),
            2: pygame.image.load(self.game.main.get_path(f"assets/playable_piece.xcf")),
            -2: pygame.image.load(self.game.main.get_path(f"assets/playable_piece_2.xcf")),
        }

        self.image = pygame.image.load(self.game.main.get_path(f"assets/{self.color}_piece.xcf"))
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.x_placement = self.game.game_board_x + 50 + self.y * 100
        self.y_placement = self.game.game_board_y + 50 + self.x * 100

        self.rect = self.image.get_rect(center=(self.x_placement, self.y_placement))

    def placement(self):
        self.number = self.game.mixed_board[self.x, self.y] if self.game.is_prepared else self.game.board_shown[self.x, self.y]

        if self.number != 0:
            self.image = self.images[self.number] if self.number != 2 else self.images[self.number] if not self.rect.collidepoint((self.game.mouse_x, self.game.mouse_y)) or self.game.main.state in ["showing_results", "resigning"] else self.images[-self.number]
            self.image.set_alpha(150) if self.number == 2 else None
            self.rect = self.image.get_rect(center=(self.x_placement, self.y_placement))

            self.game.main.screen.blit(self.image, self.rect)


class FakePiece(pygame.sprite.Sprite):
    def __init__(self, main, number):
        super().__init__()
        self.main = main
        self.number = number
        self.color = "black" if self.number in [1, 2] else "white"

        self.image = pygame.image.load(self.main.get_path(f"assets/{self.color}_piece.xcf"))
        self.rect = self.image.get_rect()

        self.x = 3 if self.number in [0, 2] else 4
        self.y = 3 if self.number in [0, 1] else 4
        self.x_placement = self.main.game.game_board_x + 50 + self.y * 100
        self.y_placement = -100
        self.y_placement_max = self.main.game.game_board_y + 50 + self.x * 100

        self.velocity = round(20 * self.main.ratio_fps)
        self.velocity_decayrate = 1 - round(0.12 * self.main.ratio_fps, 2)
        self.velocity_rate = 1.0

        self.hidden = True

    def placement(self):
        self.velocity = round(20 * self.main.ratio_fps)
        self.velocity_decayrate = 1 - round(0.11 * self.main.ratio_fps, 2)

        self.rect = self.image.get_rect(center=(self.x_placement, self.y_placement))

        if not self.hidden:
            self.main.screen.blit(self.image, self.rect)
