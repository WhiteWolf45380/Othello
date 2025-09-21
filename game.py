import numpy as np
import pygame
from random import choice
from pieces import Piece
from sound import SoundManager


class Game:

    def __init__(self, main):
        self.main = main  # passerelle avec main

        # -1 = noir et 1 = blanc

        # initialisation du plateau
        self.board = np.zeros((8, 8))  # création d'un plateau de 0 en 8 * 8

        # placement des premiers pions
        self.board[3, 3:5] = [1, -1]
        self.board[4, 3:5] = [-1, 1]

        self.board_shown = self.board.copy() # quelle position de la partie est affiché
        self.board_shown_number = 0


        self.mixed_board = np.copy(self.board)  # plateau affichant les coups possibles avec des 2
        self.valid_moves_mask = np.zeros_like(self.board, dtype=bool)  # masque des coups valides
        self.return_board = np.full(self.board.shape, "", dtype=object)  # tableau pour accumuler les directions

        self.player_turn = -1  # premier coup aux noirs
        self.ddqn_number = choice([-1, 1])  # choix aléatoire d'une couleur pour l'agent dqn
        self.player_1 = choice([-1, 1]) # joueur 1 commence avec une couleur aléatoire
        self.rewards = []  # liste des récompenses de l'agent durant la partie
        self.bot = None # choix du bot
        self.has_to_save = True

        # liste des déplacements pour le contour d'une position
        self.directions = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

        # pattern utilisé pour le choix des coups
        self.pattern = '([0-7]),([0-7])'

        # mise à zéro des scores
        self.white_score = np.sum(self.board == 1)
        self.black_score = np.sum(self.board == -1)

        # variable vérifiant qu'un joueur ne peux pas jouer
        self.cant_play = False

        # lancement de la partie
        self.not_ended = False
        self.pressed = {}  # dictionnaire des touches pressées
        self.sound_manager = SoundManager(self)  # création d'un objet sound_manager répértoriant l'ensemble des sons
        self.mouse_x, self.mouse_y = 0, 0
        self.is_prepared = False  # vérifie s'il y a besoin de faire les calculs du tour
        self.waiting_for = ''  # attend qu'un joueur joue
        self.preparation_cooldown = 0 # pour mettre un délai entre le coup du joueur et de l'ia

        # assets pygame :

        # le fond d'écran
        self.background = pygame.image.load(self.main.get_path("assets/background.xcf"))
        self.background = pygame.transform.scale(self.background, (self.main.screen_width, self.main.screen_height))
        self.background_rect = self.background.get_rect(topleft=(0, 0))

        # le plateau de jeu
        self.game_board_dict = {
            "wood": (pygame.image.load(self.main.get_path('assets/boards/wood.xcf')), (160, 82, 45), (210, 140, 80)),
            "green": (pygame.image.load(self.main.get_path('assets/boards/green.xcf')), (255, 255, 50), (255, 255, 100)),
            "ice sea": (pygame.image.load(self.main.get_path('assets/boards/ice_sea.xcf')), (30, 160, 235), (135, 200, 250)),
            "red": (pygame.image.load(self.main.get_path('assets/boards/red.xcf')), (255, 150, 50), (255, 200, 150)),
            "stone": (pygame.image.load(self.main.get_path('assets/boards/stone.xcf')), (55, 100, 155), (100, 135, 175)),
            "glass": (pygame.image.load(self.main.get_path('assets/boards/glass.xcf')), (50, 120, 190), (100, 125, 90)),
        }

        self.game_board = self.game_board_dict["wood"][0]  # plateau de jeu
        self.game_board_original = self.game_board.copy()  # copie du plateau
        self.game_board_x = (self.main.screen_width - 800) // 2  # coordonnée x du coin supérieur gauche du plateau
        self.game_board_y = (self.main.screen_height - 800) // 2  # coordonnée y du coin supérieur gauche du plateau
        self.game_board_rect = self.game_board.get_rect(center=(
        self.game_board_x + 400, self.game_board_y + 400))  # rect temporaire pour ancrer le centre du plateau
        self.game_board_width = self.game_board.get_width()  # largeur du plateau
        self.game_board_height = self.game_board.get_height()  # hauteur du plateau
        self.game_board_overlay = pygame.Surface((self.main.screen_width,
                                                  self.main.screen_height))  # création d'un voile plutot qui rend mieux que de faire un réel fondu du plateau
        self.game_board_overlay.fill((0, 0, 0))  # voile complément noir
        self.game_board_overlay_alpha = 255  # opacité du voile
        self.game_board_scale = 0.01  # proportions du plateau
        self.game_board_cooldown = 0  # cooldown pour les modifications apportées au plateau

        # coup injouable
        self.unplayable = pygame.Surface((100, 100))
        self.unplayable.fill((255, 0, 0))
        self.unplayable.set_alpha(110)
        self.unplayable_x = 0
        self.unplayable_y = 0
        self.unplayable_rect = self.unplayable.get_rect()
        self.unplayable_draw = False
        self.unplayable_cooldown = 0
        self.unplayable_cnt = 4

        # dernier coup joué
        self.last_move = pygame.Surface((100, 100))
        self.last_move.fill(self.game_board_dict[self.main.memory.datas["board"]][1])
        self.last_move.set_alpha(180)
        self.last_move_x = 0
        self.last_move_y = 0
        self.last_move_rect = self.unplayable.get_rect()

        self.returned_pieces = {} # les dernières pièces retournées

        # barre latérale
        self.sidebar_width = 75
        self.sidebar_width_min = 75
        self.sidebar_width_max = 400
        self.sidebar = pygame.Surface((self.sidebar_width, self.main.screen_height))
        self.sidebar.fill((40, 40, 40))
        self.sidebar.set_alpha(200)

        # curseur de la barre latérale
        self.sidebar_cursor = pygame.Surface((self.sidebar_width_max, self.sidebar_width_min))
        self.sidebar_cursor_rect = self.sidebar_cursor.get_rect()
        self.sidebar_cursor_x = 0
        self.sidebar_cursor_y = 0
        self.sidebar_cursor.fill((10, 10, 10))
        self.sidebar_cursor_current = ""
        self.sidebar_cursor_draw = False

        # icones et textes de la barre latérale
        self.icons = []
        self.icons_font = pygame.font.Font(self.main.get_path("assets/fonts/icons.ttf"), 40)

        # pour quitter le jeu
        self.icon_log_out = pygame.image.load(self.main.get_path("assets/icons/side_bar/log_out.xcf"))
        self.icon_log_out_x = self.sidebar_width_min / 2
        self.icon_log_out_y = self.main.screen_height - self.sidebar_width_min / 2
        self.icon_log_out_rect = self.icon_log_out.get_rect(center=(self.icon_log_out_x, self.icon_log_out_y))
        self.icon_log_out_text = self.icons_font.render("Quitter", 1, (255, 255, 255))
        self.icon_log_out_text_rect = self.icon_log_out_text.get_rect(midleft=(self.sidebar_width_min + 20, self.icon_log_out_y))
        self.icons.append((0, self.icon_log_out_y, "log out"))

        # pour revenir au choix des modes
        self.icon_play = pygame.image.load(self.main.get_path("assets/icons/side_bar/play.xcf"))
        self.icon_play_x = self.sidebar_width_min / 2
        self.icon_play_y = self.sidebar_width_min / 2
        self.icon_play_rect = self.icon_play.get_rect(center=(self.icon_play_x, self.icon_play_y))
        self.icon_play_text = self.icons_font.render("Jouer", 1, (255, 255, 255))
        self.icon_play_text_rect = self.icon_play_text.get_rect(
            midleft=(self.sidebar_width_min + 20, self.icon_play_y))
        self.icons.append((0, self.icon_play_y, "play"))

        # pour accéder aux paramètres
        self.icon_settings = pygame.image.load(self.main.get_path("assets/icons/side_bar/settings.xcf"))
        self.icon_settings_x = self.sidebar_width_min / 2
        self.icon_settings_y = self.main.screen_height - self.sidebar_width_min / 2 - self.sidebar_width_min
        self.icon_settings_rect = self.icon_play.get_rect(center=(self.icon_settings_x, self.icon_settings_y))
        self.icon_settings_text = self.icons_font.render("Paramètres", 1, (255, 255, 255))
        self.icon_settings_text_rect = self.icon_settings_text.get_rect(
            midleft=(self.sidebar_width_min + 20, self.icon_settings_y))
        self.icons.append((0, self.icon_settings_y, "settings"))

        # fond du menu de droite
        self.back_choosing_mod = pygame.Surface((self.main.screen_width - self.game_board_x - self.game_board.get_width() - 100,
                                                 self.main.screen_height - 100))
        self.back_choosing_mod_x = self.game_board_x + self.game_board.get_width() + 50
        self.back_choosing_mod_y = 50
        self.back_choosing_mod_rect = self.back_choosing_mod.get_rect(topleft=(self.back_choosing_mod_x, self.back_choosing_mod_y))
        self.back_choosing_mod.fill((25, 25, 25))
        self.back_choosing_mod.set_alpha(210)

        # assets du choix du mode

        # titre du menu
        self.choosing_mod_title_font = pygame.font.Font(self.main.get_path("assets/fonts/choosing_mod_title.ttf"), 50)
        self.choosing_mod_title = self.choosing_mod_title_font.render("Jouer à Othello", 1, (255, 255, 255))
        self.choosing_mod_title_rect = self.choosing_mod_title.get_rect(center=(self.back_choosing_mod_x
            + self.back_choosing_mod.get_width() / 2, self.back_choosing_mod_y + self.choosing_mod_title.get_height() / 2 + 10))

        # fond des modes
        self.mods_background = []

        for i in range(3):
            back = pygame.Rect(0, 0, self.back_choosing_mod.get_width() - 50, self.back_choosing_mod.get_height() / 8)
            back.center=(self.back_choosing_mod_x + self.back_choosing_mod.get_width() / 2,
                    int(self.back_choosing_mod_y + self.back_choosing_mod.get_height() / 2.9 + 1.3 * i * self.back_choosing_mod.get_height() // 8))
            self.mods_background.append(back)

        # icones et textes
        self.choosing_mod_main_font = pygame.font.Font(self.main.get_path("assets/fonts/choosing_mod_main_text.ttf"), 30)
        self.choosing_mod_second_font = pygame.font.Font(self.main.get_path("assets/fonts/choosing_mod_second_text.ttf"), 15)

        self.choosing_mod_game_icon = pygame.image.load(self.main.get_path("assets/icons/right_menu/game.xcf"))
        self.choosing_mod_game_icon_rect = self.choosing_mod_game_icon.get_rect(center=(self.back_choosing_mod_x +
            self.back_choosing_mod.get_width() / 2, self.back_choosing_mod_y + self.choosing_mod_title.get_height() +
                self.choosing_mod_game_icon.get_height() / 1.5))

        self.local_mod_icon = pygame.image.load(self.main.get_path("assets/icons/right_menu/local_mod.xcf"))
        self.local_mod_icon_rect = self.local_mod_icon.get_rect(center=(self.mods_background[0].midleft[0] + self.local_mod_icon.get_width() / 1.9,
               self.mods_background[0].midleft[1]))
        self.local_mod_main_text = self.choosing_mod_main_font.render("Jouer en local", 1, (255, 255, 255))
        self.local_mod_main_text_rect = self.local_mod_main_text.get_rect(midleft=(self.mods_background[0].midleft[0] + 75,
            self.mods_background[0].midleft[1] - 10))
        self.local_mod_second_text = self.choosing_mod_second_font.render("Jouez sur le même ordinateur chacun son tour", 1, (255, 255, 255))
        self.local_mod_second_text_rect = self.local_mod_main_text.get_rect(midleft=(self.mods_background[0].midleft[0] + 75,
            self.mods_background[0].midleft[1] + 25))

        self.bots_mod_icon = pygame.image.load(self.main.get_path("assets/icons/right_menu/bots_mod.xcf"))
        self.bots_mod_icon_rect = self.bots_mod_icon.get_rect(center=(self.mods_background[1].midleft[0] + self.bots_mod_icon.get_width() / 1.9,
               self.mods_background[1].midleft[1]))
        self.bots_mod_main_text = self.choosing_mod_main_font.render("Jouer contre des robots", 1, (255, 255, 255))
        self.bots_mod_main_text_rect = self.bots_mod_main_text.get_rect(midleft=(self.mods_background[1].midleft[0] + 75,
            self.mods_background[1].midleft[1] - 10))
        self.bots_mod_second_text = self.choosing_mod_second_font.render("Affrontez différents modèles d'IA", 1, (255, 255, 255))
        self.bots_mod_second_text_rect = self.bots_mod_main_text.get_rect(midleft=(self.mods_background[1].midleft[0] + 75,
            self.mods_background[1].midleft[1] + 25))

        self.training_mod_icon = pygame.image.load(self.main.get_path("assets/icons/right_menu/training_mod.xcf"))
        self.training_mod_icon_rect = self.training_mod_icon.get_rect(center=(self.mods_background[2].midleft[0] + self.local_mod_icon.get_width() / 1.9,
               self.mods_background[2].midleft[1]))
        self.training_mod_main_text = self.choosing_mod_main_font.render("Entrainer des robots", 1, (255, 255, 255))
        self.training_mod_main_text_rect = self.training_mod_main_text.get_rect(midleft=(self.mods_background[2].midleft[0] + 75,
            self.mods_background[2].midleft[1] - 10))
        self.training_mod_second_text = self.choosing_mod_second_font.render("Améliorez les modèles d'IA", 1, (255, 255, 255))
        self.training_mod_second_text_rect = self.training_mod_main_text.get_rect(midleft=(self.mods_background[2].midleft[0] + 75,
            self.mods_background[2].midleft[1] + 25))

        # profil des deux joueurs
        self.profil_font = pygame.font.Font(self.main.get_path("assets/fonts/profils.ttf"), 35)

        self.profil_pictures = {
            "original": {
                1: pygame.image.load(self.main.get_path("assets/profil_pictures/original_white.xcf")),
                -1: pygame.image.load(self.main.get_path("assets/profil_pictures/original_black.xcf"))
            },

            "ddqn": {
                1: pygame.image.load(self.main.get_path("assets/profil_pictures/ddqn_white.xcf")),
                -1: pygame.image.load(self.main.get_path("assets/profil_pictures/ddqn_black.xcf"))
            },

            "minmax": {
                1: pygame.image.load(self.main.get_path("assets/profil_pictures/minmax_white.xcf")),
                -1: pygame.image.load(self.main.get_path("assets/profil_pictures/minmax_black.xcf"))
            }
        }

        self.profil_player_1_picture = self.profil_pictures["original"][1]
        self.profil_player_2_picture = self.profil_pictures["original"][-1]

        self.profil_player_1_picture_rect = self.profil_player_1_picture.get_rect(topleft=(self.game_board_x, self.game_board_y + self.game_board_height + 13))
        self.profil_player_2_picture_rect = self.profil_player_2_picture.get_rect(topleft=(self.game_board_x, self.game_board_y - self.profil_player_2_picture.get_height() - 10))

        self.profils_text = {
            "player_1": self.profil_font.render("Joueur 1", 1, (255, 255, 255)),
            "player_2": self.profil_font.render("Joueur 2", 1, (255, 255, 255)),
            "minmax": self.profil_font.render("Minmax", 1, (255, 255, 255)),
            "ddqn": self.profil_font.render("Double-DQN", 1, (255, 255, 255))
        }

        self.profil_player_1_text = self.profils_text["player_1"]
        self.profil_player_1_text_rect = self.profils_text["player_1"].get_rect(topleft=(
            self.game_board_x + self.profil_player_1_picture.get_width() + 10, self.game_board_y + self.game_board.get_height() + 10))
        self.profil_player_2_text = self.profils_text["player_2"]
        self.profil_player_2_text_rect = self.profils_text["player_2"].get_rect(topleft=(
            self.game_board_x + self.profil_player_2_picture.get_width() + 10, self.game_board_y - self.profil_player_2_picture.get_height() - 10))

        # choix du bot
        self.bot_minmax_back = self.mods_background[0].copy()
        self.bot_minmax_icon = self.profil_pictures["minmax"][1]
        self.bot_minmax_icon_rect = self.bot_minmax_icon.get_rect(center=(self.local_mod_icon_rect.center[0] + 15, self.local_mod_icon_rect.center[1]))
        self.bot_minmax_text_main = self.choosing_mod_main_font.render("MinMax", 1, (255, 255, 255))
        self.bot_minmax_text_main_rect = self.bot_minmax_text_main.get_rect(midleft=(self.bot_minmax_icon_rect.midright[0] + 10, self.bot_minmax_icon_rect.midright[1] - 15))
        self.bot_minmax_text_second = self.choosing_mod_second_font.render("Tree Search", 1, (255, 255, 255))
        self.bot_minmax_text_second_rect = self.bot_minmax_text_second.get_rect(midleft=(self.bot_minmax_icon_rect.midright[0] + 10, self.bot_minmax_icon_rect.midright[1] + 15))

        self.bot_ddqn_back = self.mods_background[1].copy()
        self.bot_ddqn_icon = self.profil_pictures["ddqn"][1]
        self.bot_ddqn_icon_rect = self.bot_ddqn_icon.get_rect(center=(self.bots_mod_icon_rect.center[0] + 15, self.bots_mod_icon_rect.center[1]))
        self.bot_ddqn_text_main = self.choosing_mod_main_font.render("Double-DQN", 1, (255, 255, 255))
        self.bot_ddqn_text_main_rect = self.bot_ddqn_text_main.get_rect(midleft=(self.bot_ddqn_icon_rect.midright[0] + 10, self.bot_ddqn_icon_rect.midright[1] - 15))
        self.bot_ddqn_text_second = self.choosing_mod_second_font.render("Deep Reinforcement Learning", 1, (255, 255, 255))
        self.bot_ddqn_text_second_rect = self.bot_ddqn_text_second.get_rect(midleft=(self.bot_ddqn_icon_rect.midright[0] + 10, self.bot_ddqn_icon_rect.midright[1] + 15))

        # barre des scores
        self.scores_bar_width = 50
        self.white_score_bar = pygame.Surface((self.scores_bar_width, self.game_board.get_height()))
        self.white_score_bar.fill((255, 255, 255))
        self.black_score_bar = pygame.Surface((self.scores_bar_width, self.game_board_height * np.sum(self.board == -1) / np.sum(self.board != 0)))
        self.black_score_bar.fill((10, 10, 10))
        self.white_score_bar_rect = self.white_score_bar.get_rect(topright=(self.game_board_x - 20, self.game_board_y))
        if self.player_1 == 1:
            self.black_score_bar_rect = self.black_score_bar.get_rect(topright=(self.game_board_x - 20, self.game_board_y))
        else:
            self.black_score_bar_rect = self.black_score_bar.get_rect(bottomright=(self.game_board_x - 20, self.game_board_y + self.game_board.get_height()))
        self.black_score_bar_height = self.white_score_bar.get_height() * np.sum(self.board==-1) / np.sum(self.board != 0)

        self.scores_bar_font = pygame.font.Font(self.main.get_path("assets/fonts/scores_bar.ttf"), 30)

        self.player_1_score = self.scores_bar_font.render(f"{self.white_score if self.player_1 == 1 else self.black_score}", 1, (127, 127, 127))
        self.player_1_score_rect = self.player_1_score.get_rect(midbottom=(self.white_score_bar_rect.midbottom[0], self.white_score_bar_rect.midbottom[1] - 5))
        self.player_2_score = self.scores_bar_font.render(f"{self.black_score if self.player_1 == 1 else self.white_score}", 1, (127, 127, 127))
        self.player_2_score_rect = self.player_2_score.get_rect(midtop=(self.white_score_bar_rect.midtop[0], self.white_score_bar_rect.midtop[1] + 5))

        self.right_bottom_texts_font = pygame.font.Font(self.main.get_path("assets/fonts/right_bottom.ttf"), 17)

        # affichage du joueur du tour
        self.player_turn_font = pygame.font.Font(self.main.get_path("assets/fonts/player_turn.ttf"), 45)
        self.player_turn_text = self.player_turn_font.render("Au tour des noirs", 1, (255, 255, 255))
        self.player_turn_text_rect = self.player_turn_text.get_rect(midtop=(self.back_choosing_mod_rect.midtop[0],
                                                                            self.back_choosing_mod_rect.midtop[1] + self.player_turn_text.get_height() / 4))

        if self.player_turn == self.player_1:
            self.player_turn_picture = pygame.transform.scale(self.profil_player_1_picture, (100, 100))
        else:
            self.player_turn_picture = pygame.transform.scale(self.profil_player_1_picture, (100, 100))
        self.player_turn_picture_rect = self.player_turn_picture.get_rect(midtop=(self.player_turn_text_rect.midbottom[0],
                                                                                  self.player_turn_text_rect.midbottom[1] + self.player_turn_picture.get_height() / 4))

        # abandon
        self.white_flag = pygame.image.load(self.main.get_path("assets/icons/right_menu/white_flag.xcf"))
        self.white_flag_rect = self.white_flag.get_rect(
            center=(self.back_choosing_mod_rect.bottomleft[0] + self.white_flag.get_width(),
                    self.back_choosing_mod_rect.bottomleft[1] - self.white_flag.get_height()))
        self.white_flag_text = self.right_bottom_texts_font.render("Abandonner", 1, (255, 255, 255))
        self.white_flag_text_rect = self.white_flag_text.get_rect(center=self.white_flag_rect.midbottom)

        self.resign_back_width = self.game_board.get_width() * 3 // 8
        self.resign_back_height = self.game_board.get_height() * 1 // 8
        self.resign_back = pygame.Rect(0, 0, self.resign_back_width, self.resign_back_height)
        self.resign_back.midbottom= self.white_flag_rect.midtop
        self.resign_back_triangle = [
            (self.resign_back.midbottom[0] - 8, self.resign_back.midbottom[1]),
            (self.resign_back.midbottom[0] + 8, self.resign_back.midbottom[1]),
            (self.resign_back.midbottom[0], self.resign_back.midbottom[1] + 10)
        ]

        self.resign_text_font = pygame.font.Font(self.main.get_path("assets/fonts/resign.ttf"), 17)
        self.resign_text = self.resign_text_font.render("Voulez-vous vraiment abandonner ?", 1, (255, 255, 255))
        self.resign_text_rect = self.resign_text.get_rect(midtop=(self.resign_back.midtop[0], self.resign_back.midtop[1] + self.resign_text.get_height() * 0.3))

        self.resign_buttons_font = pygame.font.Font(self.main.get_path("assets/fonts/resign_buttons.ttf"), 25)

        self.resign_true = pygame.Rect(0, 0, self.resign_back_width / 2.75, self.resign_back_height / 2.75)
        self.resign_true.center = (self.resign_back.midleft[0] + self.resign_back_width // 4, self.resign_back.midleft[1] + self.resign_back_height // 7)
        self.resign_true_back = self.resign_true.copy()
        self.resign_true_back.center = (self.resign_true.center[0], self.resign_true.center[1] + 5)
        self.resign_true_text = self.resign_buttons_font.render("Oui", 1, (255, 255, 255))
        self.resign_true_text_rect = self.resign_true_text.get_rect(center=self.resign_true.center)

        self.resign_false = self.resign_true.copy()
        self.resign_false.center = (self.resign_back.midleft[0] + self.resign_back_width * 3 // 4, self.resign_back.midleft[1] + self.resign_back_height // 7)
        self.resign_false_back = self.resign_false.copy()
        self.resign_false_back.center = (self.resign_false.center[0], self.resign_false.center[1] + 5)
        self.resign_false_text = self.resign_buttons_font.render("Non", 1, (255, 255, 255))
        self.resign_false_text_rect = self.resign_false_text.get_rect(center=self.resign_false.center)

        # coup précédent
        self.last_one = pygame.image.load(self.main.get_path("assets/icons/right_menu/last_one_grey.xcf"))
        self.last_one_rect = self.last_one.get_rect(
            center=(self.back_choosing_mod_rect.midbottom[0] + self.last_one.get_width() * 0.07,
                    self.back_choosing_mod_rect.midbottom[1] - self.last_one.get_height()))
        self.last_one_text = self.right_bottom_texts_font.render("Précedent", 1, (255, 255, 255))
        self.last_one_text_rect = self.last_one_text.get_rect(
            center=(self.last_one_rect.midbottom[0], self.white_flag_text_rect.center[1]))

        # coup suivant
        self.next_one = pygame.image.load(self.main.get_path("assets/icons/right_menu/next_one_grey.xcf"))
        self.next_one_rect = self.next_one.get_rect(
            center=(self.back_choosing_mod_rect.bottomright[0] - self.next_one.get_width() * 0.95,
                    self.back_choosing_mod_rect.bottomright[1] - self.next_one.get_height()))
        self.next_one_text = self.right_bottom_texts_font.render("Suivant", 1, (255, 255, 255))
        self.next_one_text_rect = self.next_one_text.get_rect(
            center=(self.next_one_rect.midbottom[0], self.white_flag_text_rect.center[1]))

        # ligne de séparation
        self.separation_line = pygame.Rect(0, 0, self.back_choosing_mod.get_width() - 50, 4)
        self.separation_line.center = (self.back_choosing_mod_rect.center[0], self.white_flag_rect.top - 10)

        # suivit de la partie
        self.transitions = []
        self.transitions.append((self.board.copy(), self.player_turn, 1, 1))

        self.moves = []
        self.moves_font_size = 15
        self.moves_font = pygame.font.Font(self.main.get_path("assets/fonts/moves.ttf"), self.moves_font_size)
        self.moves_limit = 15
        self.moves_initial_x = self.back_choosing_mod_x + self.back_choosing_mod.get_width() * 0.3
        self.moves_initial_y = self.player_turn_picture_rect.bottom + self.player_turn_picture.get_height() // 2
        self.moves_x_space = self.back_choosing_mod.get_width() * 0.3
        self.moves_y_space = abs(self.separation_line.center[1] - self.player_turn_picture.get_height() / 2 - self.moves_initial_y) / self.moves_limit
        self.moves_following = pygame.Rect(0, 0, 0, 0)
        self.moves_following_index = 0

        current_move = f"{len(self.moves)}.    -"

        current_move_text = self.moves_font.render(current_move, 1, (255, 255, 255))
        current_move_text_rect = current_move_text.get_rect(midleft=(
            self.moves_initial_x + self.moves_x_space * (len(self.moves) % 2),
            self.moves_initial_y + self.moves_y_space * (len(self.moves) // 2)
        ))
        current_move_back = pygame.Rect(0, 0, current_move_text.get_width() + 3, current_move_text.get_height() + 3)
        current_move_back.center = current_move_text_rect.center
        self.moves.append((current_move_back, current_move_text, current_move_text_rect))

        # barre de défilement pour le suivit de la partie
        self.scroll_bar_height_max = self.moves_y_space * self.moves_limit
        self.scroll_bar_width = 15
        self.scroll_bar_height = self.scroll_bar_height_max * (len(self.moves) / (self.moves_limit * 2))
        self.scroll_bar_back = pygame.Rect(0, 0, self.scroll_bar_width, self.scroll_bar_height_max)
        self.scroll_bar_back.topright = (self.back_choosing_mod_rect.right - 10, self.moves_initial_y)
        self.scroll_bar = pygame.Rect(0, 0, self.scroll_bar_width, self.scroll_bar_height)
        self.scroll_bar.midtop = self.scroll_bar_back.midtop
        self.scroll_bar_y = self.scroll_bar.top
        self.scroll_bar_grabbing = False
        self.scroll_bar_delta = 0

        # chronomètres
        self.settings_timer_list = [float("inf"), 1, 2, 3, 5, 10, 15, 30, 60]
        self.timer_text_font = pygame.font.Font(self.main.get_path("assets/fonts/timer.ttf"), 30)
        self.timer_player_1 = self.settings_timer_list[self.main.memory.datas["timer"]] * 60000
        self.timer_player_2 = self.settings_timer_list[self.main.memory.datas["timer"]] * 60000
        self.timer_player_1_text = self.timer_text_font.render(
            f"{self.timer_player_1 // 60000}:{self.timer_player_1 % 60000 // 1000:02}", 1, (255, 255, 255) if self.player_1 == -1 else (0, 0, 0))
        self.timer_player_2_text = self.timer_text_font.render(
            f"{self.timer_player_2 // 60000}:{self.timer_player_2 % 60000 // 1000:02}", 1, (255, 255, 255) if self.player_1 == 1 else (0, 0, 0))
        self.timer_player_1_text_rect = self.timer_player_1_text.get_rect(topright=(self.game_board_rect.bottomright[0] - 30, self.game_board_rect.bottomright[1] + 12))
        self.timer_player_2_text_rect = self.timer_player_2_text.get_rect(bottomright=(self.game_board_rect.topright[0] - 30, self.game_board_rect.topright[1] - 10))
        self.timer_player_1_back = pygame.Rect(0, 0, self.timer_player_1_text.get_width() + 10, self.timer_player_1_text.get_height() + 5)
        self.timer_player_1_back.center = self.timer_player_1_text_rect.center
        self.timer_player_2_back = pygame.Rect(0, 0, self.timer_player_1_text.get_width() + 10, self.timer_player_1_text.get_height() + 5)
        self.timer_player_2_back.center = self.timer_player_2_text_rect.center
        self.timer_last_time = 0

        # résultat de la partie
        self.results_back_width = self.game_board.get_width() * 3 // 8
        self.results_back_height = self.game_board.get_height() * 3 // 8
        self.results_back = pygame.Rect(0, 0, self.results_back_width, self.results_back_height)
        self.results_back.center=(self.main.screen_width // 2, self.main.screen_height // 2)

        self.results_back_2 = pygame.Rect(0, 0, self.results_back_width, self.results_back_height // 2)
        self.results_back_2.midtop = self.results_back.midtop

        self.results_back_3 = pygame.Rect(0, 0, self.results_back_width, self.results_back_height // 1.7)
        self.results_back_3.center = self.results_back.center

        self.results_winner_font = pygame.font.Font(self.main.get_path("assets/fonts/results_winner.ttf"), 28)
        self.results_winner_midscore = pygame.font.Font(self.main.get_path("assets/fonts/results_winner.ttf"), 50)

        self.results_winner_text = self.results_winner_font.render(f"Les _ ont gagné", 1, (255, 255, 255))
        self.results_winner_text_rect = self.results_winner_text.get_rect(center=(self.results_back.midtop[0],
            self.results_back.midtop[1]))

        self.results_player_1_picture = pygame.transform.scale(self.profil_player_1_picture, (70, 70))
        self.results_player_2_picture = pygame.transform.scale(self.profil_player_2_picture, (70, 70))

        self.results_player_1_picture_rect = self.results_player_1_picture.get_rect(center=(self.results_back.center[0] - self.results_back_width / 4,
                                                                                            self.results_back.center[1] - self.results_back_height / 7))
        self.results_player_2_picture_rect = self.results_player_2_picture.get_rect(center=(self.results_back.center[0] + self.results_back_width / 4,
                                                                                            self.results_back.center[1] - self.results_back_height / 7))
        self.results_midscore = self.results_winner_midscore.render("-", 1, (255, 255, 255))
        self.results_midscore_rect = self.results_midscore.get_rect(center=(self.results_back.center[0], self.results_player_1_picture_rect.center[1]))

        self.results_player_1_score = self.scores_bar_font.render(f"{self.white_score if self.player_1 == 1 else self.black_score}", 1, (255, 255, 255))
        self.results_player_2_score = self.scores_bar_font.render(f"{self.black_score if self.player_1 == 1 else self.white_score}", 1, (255, 255, 255))
        self.results_player_1_score_rect = self.results_player_1_score.get_rect(midtop=self.results_player_1_picture_rect.midbottom)
        self.results_player_2_score_rect = self.results_player_2_score.get_rect(midtop=self.results_player_2_picture_rect.midbottom)

        self.results_review_width = self.results_back_width - 50
        self.results_review_height = 50
        self.results_review_back = pygame.Rect(0, 0, self.results_review_width, self.results_review_height)
        self.results_review_back.midtop = (self.results_back.center[0], self.results_back.center[1] + self.results_back_height // 8 + 5)
        self.results_review = pygame.Rect(0, 0, self.results_review_width, self.results_review_height)
        self.results_review.midtop = (self.results_back.center[0], self.results_back.center[1] + self.results_back_height // 8)

        self.results_buttons_font_1 = pygame.font.Font(self.main.get_path("assets/fonts/results_buttons.ttf"), 25)
        self.results_buttons_font_2 = pygame.font.Font(self.main.get_path("assets/fonts/results_buttons.ttf"), 15)

        self.results_review_text = self.results_buttons_font_1.render("Bilan de la partie", 1, (255, 255, 255))
        self.results_review_text_rect = self.results_review_text.get_rect(center=self.results_review.center)

        self.results_back_menu = pygame.Rect(0, 0, self.results_review_width / 2 - 5, self.results_review_height / 1.3)
        self.results_back_menu.topleft=(self.results_review_back.bottomleft[0], self.results_review_back.bottomleft[1] + 10)
        self.results_back_menu_text = self.results_buttons_font_2.render("Retour au menu", 1, (200, 200, 200))
        self.results_back_menu_text_rect = self.results_back_menu_text.get_rect(center=self.results_back_menu.center)

        self.results_new_game = pygame.Rect(0, 0, self.results_review_width / 2 - 5, self.results_review_height / 1.3)
        self.results_new_game.topright=(self.results_review_back.bottomright[0], self.results_review_back.bottomright[1] + 10)
        self.results_new_game_text = self.results_buttons_font_2.render("Nouvelle partie", 1, (200, 200, 200))
        self.results_new_game_text_rect = self.results_new_game_text.get_rect(center=self.results_new_game.center)

        # les paramètres

        # les polices
        self.settings_title_font = pygame.font.Font(self.main.get_path("assets/fonts/settings_title.ttf"), 100)
        self.settings_text_font = pygame.font.Font(self.main.get_path("assets/fonts/settings_text.ttf"), 45)
        self.settings_values_font = pygame.font.Font(self.main.get_path("assets/fonts/settings_text.ttf"), 30)

        # le titre
        self.settings_title_text = self.settings_title_font.render("Paramètres", 1, (255, 255, 255))
        self.settings_title_text_rect = self.settings_title_text.get_rect(center=(self.main.screen_width / 2, self.main.screen_height / 11))

        # les fps
        self.settings_fps_text = self.settings_text_font.render("Fps : ", 1, (255, 255, 255))
        self.settings_fps_text_rect = self.settings_fps_text.get_rect(midleft=(self.sidebar_width_max - 100 * self.main.screen_width / 1920, self.main.screen_height * 320 / 1080))
        self.settings_fps_point_radius = 25 * self.main.screen_height / 1080
        self.settings_fps_point_x = self.settings_fps_text_rect.midleft[0] + 275 * self.main.screen_width / 1920
        self.settings_fps_bar = pygame.Rect(0, 0, self.main.screen_width * 275 / 1920, 40 * self.main.screen_height / 1080)
        self.settings_fps_bar.midleft =  (self.settings_fps_point_x - self.settings_fps_point_radius, self.settings_fps_text_rect.center[1] + 5 * self.main.screen_height / 1080)
        self.settings_fps_point_grabbing = False
        self.settings_fps_point_delta = 0
        self.settings_fps_value_text = self.settings_values_font.render(str(self.main.memory.datas["fps_max"]), 1, (255, 255, 255))
        self.settings_fps_value_text_rect = self.settings_fps_value_text.get_rect(center=(self.settings_fps_bar.midright[0] + 50, self.settings_fps_bar.center[1]))
        self.settings_fps_value_back = pygame.Rect(0, 0, self.settings_fps_value_text.get_height() +  5, self.settings_fps_value_text.get_height())
        self.settings_fps_value_back.center = self.settings_fps_value_text_rect.center
        self.settings_fps_point_x += (self.main.memory.datas["fps_max"] - 10) * (self.settings_fps_bar.right - 2 * self.settings_fps_point_radius - self.settings_fps_bar.left) // 70

        # le volume sonore
        self.settings_volume_text = self.settings_text_font.render("Volume : ", 1, (255, 255, 255))
        self.settings_volume_text_rect = self.settings_volume_text.get_rect(midleft=(self.sidebar_width_max - 100 * self.main.screen_width / 1920, self.main.screen_height * (320 + 90 * 1) / 1080))
        self.settings_volume_point_radius = 25 * self.main.screen_height / 1080
        self.settings_volume_point_x = self.settings_volume_text_rect.midleft[0] + 275 * self.main.screen_width / 1920
        self.settings_volume_bar = pygame.Rect(0, 0, self.main.screen_width * 275 / 1920, 40 * self.main.screen_height / 1080)
        self.settings_volume_bar.midleft = (self.settings_volume_point_x - self.settings_volume_point_radius, self.settings_volume_text_rect.center[1] + 5 * self.main.screen_height / 1080)
        self.settings_volume_point_grabbing = False
        self.settings_volume_point_delta = 0
        self.settings_volume_value_text = self.settings_values_font.render(str(self.main.memory.datas["volume"]), 1, (255, 255, 255))
        self.settings_volume_value_text_rect = self.settings_volume_value_text.get_rect(center=(self.settings_volume_bar.midright[0] + 50, self.settings_volume_bar.center[1]))
        self.settings_volume_value_back = pygame.Rect(0, 0, self.settings_volume_value_text.get_height() + 5, self.settings_volume_value_text.get_height())
        self.settings_volume_value_back.center = self.settings_volume_value_text_rect.center
        self.settings_volume_point_x += (self.main.memory.datas["volume"]) * (self.settings_volume_bar.right - 2 * self.settings_volume_point_radius - self.settings_volume_bar.left) // 100

        # le temps par joueur
        self.settings_timer_text = self.settings_text_font.render(f"Pendule : ", 1, (255, 255, 255))
        self.settings_timer_text_rect = self.settings_timer_text.get_rect(midleft=(self.sidebar_width_max - 100 * self.main.screen_width / 1920, self.main.screen_height *  (320 + 90 * 2) / 1080))
        self.settings_timer_value_text = self.settings_values_font.render(f'{self.settings_timer_list[self.main.memory.datas["timer"]]}{":00" if self.main.memory.datas["timer"] > 0 else ""}', 1, (255, 255, 255))
        self.settings_timer_value_text_rect = self.settings_timer_value_text.get_rect(center=(self.settings_volume_bar.center[0] + 10, self.settings_timer_text_rect.center[1]))
        self.settings_timer_value_back = pygame.Rect(0, 0, 130, self.settings_timer_value_text.get_height() + 5)
        self.settings_timer_value_back.center = self.settings_timer_value_text_rect.center
        self.settings_timer_plus_back = pygame.Rect(0, 0, 130 / 2, self.settings_timer_value_text.get_height() + 5)
        self.settings_timer_plus_back.midleft = (self.settings_timer_value_back.right + 5, self.settings_timer_value_back.midright[1])
        self.settings_timer_minus_back = pygame.Rect(0, 0, 130 / 2, self.settings_timer_value_text.get_height() + 5)
        self.settings_timer_minus_back.midright = (self.settings_timer_value_back.left - 5, self.settings_timer_value_back.midleft[1])

        # le plateau de jeu
        self.settings_board_picture = pygame.transform.scale(self.game_board, (self.game_board_width / 2, self.game_board_height / 2))
        self.settings_board_picture_rect = self.settings_board_picture.get_rect(topleft=(self.game_board_x + 6 / 8 * self.game_board.get_width(), self.game_board_y + 4 / 8 * self.game_board.get_height()))
        self.settings_board_buttons = {}

        # désactivation de l'entraînement pour l'éxécutable
        """""""""
        self.disabled_font = pygame.font.Font(self.main.get_path("assets/fonts/disabled.ttf"), 60)
        self.disabled_text = self.disabled_font.render("Fontionnalité désactivée", 1, (230, 0, 0))
        self.disabled_rect = self.disabled_text.get_rect(center=(self.main.screen_width / 2, 120))
        self.disabled_alpha = 0
        self.disabled_text.set_alpha(self.disabled_alpha)
        self.disabled_up = True
        self.disabled_showing = False
        """""""""

        for i, name in enumerate(self.game_board_dict.keys()):
            text = self.settings_values_font.render(name, 1, (255, 255, 255))
            text_rect = text.get_rect(center=(((self.settings_board_picture_rect.center[0] - 180) + i % 3 * 180), self.settings_board_picture_rect.top - 200 if i < 3 else self.settings_board_picture_rect.top - 90))
            back_0 = pygame.Rect(0, 0, 126, text.get_height() + 26)
            back_0.center = text_rect.center
            back_1 = pygame.Rect(0, 0, 120, text.get_height() + 20)
            back_1.center = back_0.center

            self.settings_board_buttons[name] = (text, text_rect, back_0, back_1)

        # ensemble des pièces
        self.all_pieces = pygame.sprite.Group()
        for i in range(8):
            for j in range(8):
                piece = Piece(self, 0, i, j)
                self.all_pieces.add(piece)

    # fonction gérant l'affichage et la mise à jour des éléments liés au jeu othello
    def game_update(self):
        if self.main.state != "settings":
            self.main.screen.blit(self.game_board, (self.game_board_x, self.game_board_y)) # le plateau

            self.black_score = np.sum(self.board == -1)  # calcul du score noir
            self.white_score = np.sum(self.board == 1)  # calcul du score blanc

            # la barre de suivit des scores
            self.black_score_bar_height = self.white_score_bar.get_height() * np.sum(self.board == -1) / np.sum(self.board != 0)
            if self.black_score_bar.get_height() < self.black_score_bar_height:
                self.black_score_bar = pygame.transform.scale(self.black_score_bar, (self.scores_bar_width, min(self.black_score_bar_height, self.black_score_bar.get_height() + 10)))
            elif self.black_score_bar.get_height() > self.black_score_bar_height:
                self.black_score_bar = pygame.transform.scale(self.black_score_bar, (self.scores_bar_width, max(self.black_score_bar_height, self.black_score_bar.get_height() - 10)))

            if self.player_1 == 1:
                self.black_score_bar_rect = self.black_score_bar.get_rect(
                    topright=(self.game_board_x - 20, self.game_board_y))
            else:
                self.black_score_bar_rect = self.black_score_bar.get_rect(
                    bottomright=(self.game_board_x - 20, self.game_board_y + self.game_board.get_height()))
            self.black_score_bar_height = self.white_score_bar.get_height() * np.sum(self.board == -1) / np.sum(
                self.board != 0)

            self.player_1_score = self.scores_bar_font.render(
                f"{self.white_score if self.player_1 == 1 else self.black_score}", 1, (127, 127, 127))
            self.player_1_score_rect = self.player_1_score.get_rect(
                midbottom=(self.white_score_bar_rect.midbottom[0], self.white_score_bar_rect.midbottom[1] - 5))
            self.player_2_score = self.scores_bar_font.render(
                f"{self.black_score if self.player_1 == 1 else self.white_score}", 1, (127, 127, 127))
            self.player_2_score_rect = self.player_2_score.get_rect(
                midtop=(self.white_score_bar_rect.midtop[0], self.white_score_bar_rect.midtop[1] + 5))

            self.main.screen.blit(self.white_score_bar, self.white_score_bar_rect)
            self.main.screen.blit(self.black_score_bar, self.black_score_bar_rect)
            self.main.screen.blit(self.player_1_score, self.player_1_score_rect)
            self.main.screen.blit(self.player_2_score, self.player_2_score_rect)

            # profil des deux joueurs
            self.main.screen.blit(self.profil_player_1_picture, self.profil_player_1_picture_rect)
            self.main.screen.blit(self.profil_player_2_picture, self.profil_player_2_picture_rect)

            self.main.screen.blit(self.profil_player_1_text, self.profil_player_1_text_rect)
            self.main.screen.blit(self.profil_player_2_text, self.profil_player_2_text_rect)

            self.main.screen.blit(self.back_choosing_mod, self.back_choosing_mod_rect) # fond du menu de droite

            if self.main.state in ["playing", "reviewing", "showing_results", "resigning"]:
                # affichage du dernier coup joué
                if np.sum(self.board_shown == 0) < 60:
                    self.main.screen.blit(self.last_move, self.last_move_rect)
                    for piece in self.returned_pieces.values():
                        self.main.screen.blit(piece[0], piece[1])

                if self.main.state in ["playing", "reviewing", "resigning"]:  # entrain de jouer ou de revoir la partie

                    # coup non jouable
                    if self.main.state == "playing":
                        if self.unplayable_draw and pygame.time.get_ticks() - 250 >= self.unplayable_cooldown:
                            self.unplayable_draw = False
                            self.unplayable_cooldown = pygame.time.get_ticks()
                        elif pygame.time.get_ticks() - 250 >= self.unplayable_cooldown and self.unplayable_cnt < 3:
                            self.unplayable_draw = True
                            self.unplayable_cnt += 1
                            self.unplayable_cooldown = pygame.time.get_ticks()
                        elif self.unplayable_draw:
                            self.main.screen.blit(self.unplayable, self.unplayable_rect)

                        if not self.is_prepared and (pygame.time.get_ticks() - 800 >= self.preparation_cooldown if self.waiting_for == 'human' and self.main.mod == 2 else True):
                            # calcul des coups jouables
                            self.mixed_board, self.return_board = self.turn()

                            if self.not_ended and np.sum(self.mixed_board == 2)>0:  # uniquement entrain de jouer
                                # tour des ia
                                if self.waiting_for == 'ddqn':
                                    self.ddqn_turn(self.mixed_board, self.return_board)
                                elif self.waiting_for == 'old_dqn':
                                    self.old_dqn_turn(self.mixed_board, self.return_board)
                                elif self.waiting_for == 'minmax':
                                    self.minmax_turn(self.return_board)

                # affichage du joueur du tour
                if self.player_turn == -1:
                    self.player_turn_text = self.player_turn_font.render("Au tour des noirs", 1, (255, 255, 255))
                else:
                    self.player_turn_text = self.player_turn_font.render("Au tour des blancs", 1, (255, 255, 255))

                self.player_turn_text_rect = self.player_turn_text.get_rect(
                    midtop=(self.back_choosing_mod_rect.midtop[0],
                            self.back_choosing_mod_rect.midtop[1] + self.player_turn_text.get_height() / 4))
                self.main.screen.blit(self.player_turn_text, self.player_turn_text_rect)
                self.player_turn_picture = pygame.transform.scale(self.profil_player_1_picture, (
                    100, 100)) if self.player_turn == self.player_1 else pygame.transform.scale(
                    self.profil_player_2_picture, (100, 100))
                self.main.screen.blit(self.player_turn_picture, self.player_turn_picture_rect)

                # affichage de la barre de défilement
                self.scroll_bar_height = self.scroll_bar_height_max * min(
                    1 / max((len(self.moves) if len(self.moves) % 2 == 0 else len(self.moves) - 1) / (self.moves_limit * 2), 1), 1)

                if self.scroll_bar_grabbing and self.scroll_bar_y != self.mouse_y + self.scroll_bar_delta:
                    self.scroll_bar_y = min(max(self.mouse_y + self.scroll_bar_delta, self.scroll_bar_back.top),
                                            self.scroll_bar_back.bottom - self.scroll_bar_height)

                self.scroll_bar = pygame.Rect(0, 0, self.scroll_bar_width, self.scroll_bar_height)
                self.scroll_bar.midtop = (self.scroll_bar_back.midtop[0], self.scroll_bar_y)
                pygame.draw.rect(self.main.screen, (250, 250, 250), self.scroll_bar_back)
                if self.scroll_bar_grabbing:
                    pygame.draw.rect(self.main.screen, (100, 100, 100), self.scroll_bar)
                else:
                    pygame.draw.rect(self.main.screen, (125, 125, 125), self.scroll_bar)

                # affichage du suivit de la partie
                y_difference = abs(self.scroll_bar_back.top - self.scroll_bar.top) * self.moves_limit * self.moves_y_space / self.scroll_bar_height
                for index, move in enumerate(self.moves):
                    if self.moves_initial_y <= move[2].midleft[1] - y_difference <= self.moves_initial_y + self.moves_limit * self.moves_y_space:
                        move[0].center = (move[2].center[0], move[2].center[1] - y_difference)

                        if self.board_shown_number == index:
                            pygame.draw.rect(self.main.screen, (75, 75, 75), move[0], border_radius=5)
                        else:
                            pygame.draw.rect(self.main.screen, (50, 50, 50), move[0], border_radius=5)

                        if self.main.state in ["playing", "reviewing"] and move[0].collidepoint((self.mouse_x, self.mouse_y)):
                            self.moves_following = move[0].copy()
                            self.moves_following_index = index
                            pygame.draw.rect(self.main.screen, (90, 90, 90), self.moves_following, border_radius=5)
                        self.main.screen.blit(move[1], (move[2][0], move[2][1] - y_difference))

                # ligne de séparaton avec les boutons du bas du menu
                pygame.draw.rect(self.main.screen, (200, 200, 200), self.separation_line, border_radius=15)

                if self.main.mod == 1 and self.timer_player_1 != float("inf") or self.main.state == "choosing_mod":
                    # les chronomètres
                    if self.main.state in ["playing", "reviewing", "resigning"] and self.not_ended and self.settings_timer_list[self.main.memory.datas["timer"]] > 0:
                        if self.player_1 == self.player_turn:
                            self.timer_player_1 = max(0, self.timer_player_1 - (pygame.time.get_ticks() - self.timer_last_time))
                            self.timer_player_1_back = pygame.Rect(0, 0, self.timer_player_1_text.get_width() + 10, self.timer_player_1_text.get_height() + 5)
                            self.timer_player_1_back.center = self.timer_player_1_text_rect.center
                            if self.timer_player_1 == 0:
                                self.do_results(resign=True)
                        else:
                            self.timer_player_2 = max(0, self.timer_player_2 - (pygame.time.get_ticks() - self.timer_last_time))
                            self.timer_player_2_back = pygame.Rect(0, 0, self.timer_player_1_text.get_width() + 10, self.timer_player_1_text.get_height() + 5)
                            self.timer_player_2_back.center = self.timer_player_2_text_rect.center
                            if self.timer_player_2 == 0:
                                self.do_results(resign=True)
                        self.timer_last_time = pygame.time.get_ticks()
                        self.timer_player_1_text = self.timer_text_font.render(f"{self.timer_player_1 // 60000}:{self.timer_player_1 % 60000 // 1000:02}", 1, (255, 255, 255) if self.player_1 == -1 else (0, 0, 0))
                        self.timer_player_2_text = self.timer_text_font.render(f"{self.timer_player_2 // 60000}:{self.timer_player_2 % 60000 // 1000:02}", 1, (255, 255, 255) if self.player_1 == 1 else (0, 0, 0))
                        self.timer_player_1_text_rect = self.timer_player_1_text.get_rect(topright=(self.game_board_rect.bottomright[0] - 30,self.game_board_rect.bottomright[1] + 10))
                        self.timer_player_2_text_rect = self.timer_player_2_text.get_rect(bottomright=(self.game_board_rect.topright[0] - 30,self.game_board_rect.topright[1] - 10))
                    pygame.draw.rect(self.main.screen, ((205, 205, 205) if self.player_turn == -1 else (255, 255, 255)) if self.player_1 == 1 else ((50, 50, 50) if self.player_turn == 1 else (0, 0, 0)), self.timer_player_1_back, border_radius=5)
                    pygame.draw.rect(self.main.screen, ((205, 205, 205) if self.player_turn == -1 else (255, 255, 255)) if self.player_1 == -1 else ((50, 50, 50) if self.player_turn == 1 else (0, 0, 0)), self.timer_player_2_back, border_radius=5)
                    self.main.screen.blit(self.timer_player_1_text, self.timer_player_1_text_rect)
                    self.main.screen.blit(self.timer_player_2_text, self.timer_player_2_text_rect)

                if self.main.state in ["playing", "resigning"] or self.main.state == "reviewing" and self.not_ended:
                    # boutton d'abandon
                    if self.white_flag_rect.collidepoint((self.mouse_x, self.mouse_y)):
                        self.white_flag = pygame.image.load(self.main.get_path("assets/icons/right_menu/white_flag.xcf"))
                    else:
                        self.white_flag = pygame.image.load(self.main.get_path("assets/icons/right_menu/grey_flag.xcf"))

                    self.main.screen.blit(self.white_flag, self.white_flag_rect)
                    self.main.screen.blit(self.white_flag_text, self.white_flag_text_rect)

                # boutton de retour à la position précédente
                if self.last_one_rect.collidepoint((self.mouse_x, self.mouse_y)) and self.main.state != "resigning":
                    self.last_one = pygame.image.load(self.main.get_path("assets/icons/right_menu/last_one_white.xcf"))
                else:
                    self.last_one = pygame.image.load(self.main.get_path("assets/icons/right_menu/last_one_grey.xcf"))

                self.main.screen.blit(self.last_one, self.last_one_rect)
                self.main.screen.blit(self.last_one_text, self.last_one_text_rect)

                # boutton de passage à la position suivante
                if self.next_one_rect.collidepoint((self.mouse_x, self.mouse_y)) and self.main.state != "resigning":
                    self.next_one = pygame.image.load(self.main.get_path("assets/icons/right_menu/next_one_white.xcf"))
                else:
                    self.next_one = pygame.image.load(self.main.get_path("assets/icons/right_menu/next_one_grey.xcf"))

                self.main.screen.blit(self.next_one, self.next_one_rect)
                self.main.screen.blit(self.next_one_text, self.next_one_text_rect)

            # les pièces
            for piece in self.all_pieces:
                piece.placement()

            if self.main.state in ["choosing_mod", "choosing_bot"]: # dans le menu de choix du mode
                self.main.screen.blit(self.choosing_mod_title, self.choosing_mod_title_rect) # titre du menu
                self.main.screen.blit(self.choosing_mod_game_icon, self.choosing_mod_game_icon_rect) # logo du jeu sous le titre

                if self.main.state == "choosing_mod":

                    # fonds des choix de mode de jeu
                    for back in self.mods_background:
                        if back.collidepoint(self.mouse_x, self.mouse_y):
                            pygame.draw.rect(self.main.screen, (20, 20, 20), back, border_radius=8)
                        else:
                            pygame.draw.rect(self.main.screen, (15, 15, 15), back, border_radius=8)

                    # différents modes de jeu
                    self.main.screen.blit(self.local_mod_icon, self.local_mod_icon_rect)
                    self.main.screen.blit(self.local_mod_main_text, self.local_mod_main_text_rect)
                    self.main.screen.blit(self.local_mod_second_text, self.local_mod_second_text_rect)
                    self.main.screen.blit(self.bots_mod_icon, self.bots_mod_icon_rect)
                    self.main.screen.blit(self.bots_mod_main_text, self.bots_mod_main_text_rect)
                    self.main.screen.blit(self.bots_mod_second_text, self.bots_mod_second_text_rect)
                    self.main.screen.blit(self.training_mod_icon, self.training_mod_icon_rect)
                    self.main.screen.blit(self.training_mod_main_text, self.training_mod_main_text_rect)
                    self.main.screen.blit(self.training_mod_second_text, self.training_mod_second_text_rect)
                else:
                    if self.main.mod == 2:
                        if self.bot_minmax_back.collidepoint(self.mouse_x, self.mouse_y):
                            pygame.draw.rect(self.main.screen, (20, 20, 20), self.bot_minmax_back, border_radius=8)
                        else:
                            pygame.draw.rect(self.main.screen, (15, 15, 15), self.bot_minmax_back, border_radius=8)
                        self.main.screen.blit(self.bot_minmax_icon, self.bot_minmax_icon_rect)
                        self.main.screen.blit(self.bot_minmax_text_main, self.bot_minmax_text_main_rect)
                        self.main.screen.blit(self.bot_minmax_text_second, self.bot_minmax_text_second_rect)

                    if self.bot_ddqn_back.collidepoint(self.mouse_x, self.mouse_y):
                        pygame.draw.rect(self.main.screen, (20, 20, 20), self.bot_ddqn_back, border_radius=8)
                    else:
                        pygame.draw.rect(self.main.screen, (15, 15, 15), self.bot_ddqn_back, border_radius=8)
                    self.main.screen.blit(self.bot_ddqn_icon, self.bot_ddqn_icon_rect)
                    self.main.screen.blit(self.bot_ddqn_text_main, self.bot_ddqn_text_main_rect)
                    self.main.screen.blit(self.bot_ddqn_text_second, self.bot_ddqn_text_second_rect)

            if self.main.state == "resigning":
                pygame.draw.polygon(self.main.screen, (50, 50, 50), self.resign_back_triangle)
                pygame.draw.rect(self.main.screen, (50, 50, 50), self.resign_back, border_radius=10)

                self.main.screen.blit(self.resign_text, self.resign_text_rect)

                if self.resign_true.collidepoint((self.mouse_x, self.mouse_y)):
                    pygame.draw.rect(self.main.screen, (0, 122, 0), self.resign_true_back, border_radius=5)
                    pygame.draw.rect(self.main.screen, (0, 224, 0), self.resign_true, border_radius=5)
                else:
                    pygame.draw.rect(self.main.screen, (0, 102, 0), self.resign_true_back, border_radius=5)
                    pygame.draw.rect(self.main.screen, (0, 204, 0), self.resign_true, border_radius=5)

                if self.resign_false.collidepoint((self.mouse_x, self.mouse_y)):
                    pygame.draw.rect(self.main.screen, (122, 0, 0), self.resign_false_back, border_radius=5)
                    pygame.draw.rect(self.main.screen, (224, 0, 0), self.resign_false, border_radius=5)
                else:
                    pygame.draw.rect(self.main.screen, (102, 0, 0), self.resign_false_back, border_radius=5)
                    pygame.draw.rect(self.main.screen, (204, 0, 0), self.resign_false, border_radius=5)

                self.main.screen.blit(self.resign_true_text, self.resign_true_text_rect)
                self.main.screen.blit(self.resign_false_text, self.resign_false_text_rect)


            if self.main.state == "showing_results":
                # fenêtre de fin de partie
                pygame.draw.rect(self.main.screen, (40, 40, 40), self.results_back, border_radius=20)
                pygame.draw.rect(self.main.screen, (20, 20, 20), self.results_back_2, border_radius=20)
                pygame.draw.rect(self.main.screen, (40, 40, 40), self.results_back_3)

                # gagnant
                self.main.screen.blit(self.results_winner_text, self.results_winner_text_rect)

                # images de profil des deux adversaires
                self.main.screen.blit(self.results_player_1_picture, self.results_player_1_picture_rect)
                self.main.screen.blit(self.results_player_2_picture, self.results_player_2_picture_rect)

                # séparation entre les deux images
                self.main.screen.blit(self.results_midscore, self.results_midscore_rect)

                # scores des deux joueurs
                self.main.screen.blit(self.results_player_1_score, self.results_player_1_score_rect)
                self.main.screen.blit(self.results_player_2_score, self.results_player_2_score_rect)

                # boutton pour analyser la partie
                if self.results_review.collidepoint((self.mouse_x, self.mouse_y)):
                    pygame.draw.rect(self.main.screen, (0, 122, 0), self.results_review_back, border_radius=15)
                    pygame.draw.rect(self.main.screen, (0, 224, 0), self.results_review, border_radius=15)
                else:
                    pygame.draw.rect(self.main.screen, (0, 102, 0), self.results_review_back, border_radius=15)
                    pygame.draw.rect(self.main.screen, (0, 204, 0), self.results_review, border_radius=15)

                self.main.screen.blit(self.results_review_text, self.results_review_text_rect)

                # boutton pour revenir au menu de choix du mode
                if self.results_back_menu.collidepoint((self.mouse_x, self.mouse_y)):
                    pygame.draw.rect(self.main.screen, (80, 80, 80), self.results_back_menu, border_radius=10)
                else:
                    pygame.draw.rect(self.main.screen, (60, 60, 60), self.results_back_menu, border_radius=10)

                # boutton pour relancer une partie
                if self.results_new_game.collidepoint((self.mouse_x, self.mouse_y)):
                    pygame.draw.rect(self.main.screen, (80, 80, 80), self.results_new_game, border_radius=10)
                else:
                    pygame.draw.rect(self.main.screen, (60, 60, 60), self.results_new_game, border_radius=10)

                self.main.screen.blit(self.results_back_menu_text, self.results_back_menu_text_rect)
                self.main.screen.blit(self.results_new_game_text, self.results_new_game_text_rect)

        else: # dans les paramètres
            self.main.screen.blit(self.main.frozen_screen, (0, 0))
            self.main.screen.blit(self.game_board_overlay, (0, 0))

            self.main.screen.blit(self.settings_title_text, self.settings_title_text_rect)

            # fps
            if self.settings_fps_point_grabbing and self.settings_fps_point_x + self.settings_fps_point_delta != self.mouse_x:
                self.settings_fps_point_x = min(max(self.mouse_x + self.settings_fps_point_delta,
                                                    self.settings_fps_bar.left + self.settings_fps_point_radius),
                                                    self.settings_fps_bar.right - self.settings_fps_point_radius)
                self.settings_fps_value_text = self.settings_values_font.render(str(self.main.memory.datas["fps_max"]), 1, (255, 255, 255))
                self.settings_fps_value_text_rect = self.settings_fps_value_text.get_rect(center=(self.settings_fps_bar.midright[0] + 50, self.settings_fps_bar.center[1]))
                self.settings_fps_value_back.width = self.settings_fps_value_text.get_width() + 5
                self.settings_fps_value_back.center = self.settings_fps_value_text_rect.center
                self.main.memory.datas["fps_max"] = int(((self.settings_fps_point_x - (self.settings_fps_bar.left + self.settings_fps_point_radius)) * 70 //
                                                ((self.settings_fps_bar.right - self.settings_fps_point_radius) - (self.settings_fps_bar.left + self.settings_fps_point_radius))) + 10)
            self.main.screen.blit(self.settings_fps_text, self.settings_fps_text_rect)
            pygame.draw.rect(self.main.screen, (255, 255, 255), self.settings_fps_bar, border_radius=20)
            pygame.draw.circle(self.main.screen, (80, 80, 80), (self.settings_fps_point_x, self.settings_fps_bar.center[1]), self.settings_fps_point_radius)
            pygame.draw.circle(self.main.screen, (100, 100, 100), (self.settings_fps_point_x, self.settings_fps_bar.center[1]), self.settings_fps_point_radius - 5)
            pygame.draw.rect(self.main.screen, (60, 60, 60), self.settings_fps_value_back, border_radius=5)
            self.main.screen.blit(self.settings_fps_value_text, self.settings_fps_value_text_rect)

            # volume sonore
            if self.settings_volume_point_grabbing and self.settings_volume_point_x + self.settings_volume_point_delta != self.mouse_x:
                self.settings_volume_point_x = min(max(self.mouse_x + self.settings_volume_point_delta,
                                                    self.settings_volume_bar.left + self.settings_volume_point_radius),
                                                    self.settings_volume_bar.right - self.settings_volume_point_radius)
                self.settings_volume_value_text = self.settings_values_font.render(str(self.main.memory.datas["volume"]), 1, (255, 255, 255))
                self.settings_volume_value_text_rect = self.settings_volume_value_text.get_rect(center=(self.settings_volume_bar.midright[0] + 50, self.settings_volume_bar.center[1]))
                self.settings_volume_value_back.width = self.settings_volume_value_text.get_width() + 5
                self.settings_volume_value_back.center = self.settings_volume_value_text_rect.center
                self.main.memory.datas["volume"] = int(((self.settings_volume_point_x - (self.settings_volume_bar.left + self.settings_volume_point_radius)) * 100 //
                                                   ((self.settings_volume_bar.right - self.settings_volume_point_radius) - (self.settings_volume_bar.left + self.settings_volume_point_radius))))
            self.main.screen.blit(self.settings_volume_text, self.settings_volume_text_rect)
            pygame.draw.rect(self.main.screen, (255, 255, 255), self.settings_volume_bar, border_radius=20)
            pygame.draw.circle(self.main.screen, (80, 80, 80), (self.settings_volume_point_x, self.settings_volume_bar.center[1]), self.settings_volume_point_radius)
            pygame.draw.circle(self.main.screen, (100, 100, 100), (self.settings_volume_point_x, self.settings_volume_bar.center[1]), self.settings_volume_point_radius - 5)
            pygame.draw.rect(self.main.screen, (60, 60, 60), self.settings_volume_value_back, border_radius=5)
            self.main.screen.blit(self.settings_volume_value_text, self.settings_volume_value_text_rect)

            # la pendule
            self.main.screen.blit(self.settings_timer_text, self.settings_timer_text_rect)

            pygame.draw.rect(self.main.screen, (50, 50, 50), self.settings_timer_value_back)
            self.main.screen.blit(self.settings_timer_value_text, self.settings_timer_value_text_rect)

            pygame.draw.rect(self.main.screen, (70, 70, 70) if self.settings_timer_minus_back.collidepoint((self.mouse_x, self.mouse_y)) else (50, 50, 50), self.settings_timer_minus_back, border_top_left_radius=15, border_bottom_left_radius=15)
            pygame.draw.line(self.main.screen, (200, 200, 200), (self.settings_timer_minus_back.left + 21,
                self.settings_timer_minus_back.midleft[1]), (self.settings_timer_minus_back.right - 21, self.settings_timer_minus_back.midright[1]), 3)
            pygame.draw.rect(self.main.screen, (70, 70, 70) if self.settings_timer_plus_back.collidepoint((self.mouse_x, self.mouse_y)) else (50, 50, 50), self.settings_timer_plus_back, border_top_right_radius=15, border_bottom_right_radius=15)
            pygame.draw.line(self.main.screen, (200, 200, 200), (self.settings_timer_plus_back.left + 21, self.settings_timer_plus_back.midleft[1]),
                (self.settings_timer_plus_back.right - 21, self.settings_timer_plus_back.midright[1]), 3)
            pygame.draw.line(self.main.screen, (200, 200, 200), (self.settings_timer_plus_back.midtop[0], self.settings_timer_plus_back.midtop[1] + 12),
                (self.settings_timer_plus_back.midbottom[0], self.settings_timer_plus_back.midbottom[1] - 12), 3)

            # le plateau de jeu
            self.settings_board_picture = pygame.transform.scale(self.game_board_dict[self.main.memory.datas["board"]][0], (self.game_board_width / 2, self.game_board_height / 2))
            self.main.screen.blit(self.settings_board_picture, self.settings_board_picture_rect)

            for button in self.settings_board_buttons.values():
                pygame.draw.rect(self.main.screen, (50, 50, 50) if button[2].collidepoint((self.mouse_x, self.mouse_y)) else (35, 35, 35), button[2], border_radius=17)
                pygame.draw.rect(self.main.screen, (70, 70, 70) if button[2].collidepoint((self.mouse_x, self.mouse_y)) else (55, 55, 55), button[3], border_radius=15)
                self.main.screen.blit(button[0], button[1])

        # volet latéral à gauche
        if self.mouse_x <= self.sidebar_width < self.sidebar_width_max:
            if self.has_to_save and self.main.state != "settings":
                self.main.frozen_screen = self.main.screen.copy()
                self.has_to_save = False

            self.sidebar_width = self.sidebar_width_max
            self.sidebar = pygame.transform.scale(self.sidebar, (self.sidebar_width, self.main.screen_height))
        elif self.mouse_x > self.sidebar_width > self.sidebar_width_min:
            self.has_to_save = True
            self.sidebar_width = self.sidebar_width_min
            self.sidebar = pygame.transform.scale(self.sidebar, (self.sidebar_width, self.main.screen_height))

        self.main.screen.blit(self.sidebar, (0, 0))

        if self.mouse_x <= self.sidebar_width:
            # icones du volet latéral
            for icon in self.icons:
                temp = pygame.Surface((self.sidebar_width_max, self.sidebar_width_min))
                temp_rect = temp.get_rect(topleft=(0, icon[1] - self.sidebar_width_min / 2))
                if temp_rect.collidepoint((self.mouse_x, self.mouse_y)):
                    self.sidebar_cursor_draw = True
                    self.sidebar_cursor_y = icon[1]
                    self.sidebar_cursor_current = icon[2]
                    self.sidebar_cursor_rect = self.sidebar_cursor.get_rect(
                        midleft=(self.sidebar_cursor_x, self.sidebar_cursor_y))
                    break
                else:
                    self.sidebar_cursor_draw = False

            self.main.screen.blit(self.sidebar_cursor, self.sidebar_cursor_rect) if self.sidebar_cursor_draw else None

            self.main.screen.blit(self.icon_log_out_text, self.icon_log_out_text_rect)
            self.main.screen.blit(self.icon_play_text, self.icon_play_text_rect)
            self.main.screen.blit(self.icon_settings_text, self.icon_settings_text_rect)

        self.main.screen.blit(self.icon_log_out, self.icon_log_out_rect)  # icone pour quitter le jeu
        self.main.screen.blit(self.icon_play, self.icon_play_rect)  # icone pour retourner au choix du mode
        self.main.screen.blit(self.icon_settings, self.icon_settings_rect)  # icone pour accéder aux paramètres

        """""""""
        if self.disabled_showing:
            if self.disabled_up:
                self.disabled_alpha = min(self.disabled_alpha + 8 * self.main.ratio_fps, 255)
                self.disabled_text.set_alpha(self.disabled_alpha)

                if self.disabled_alpha == 255:
                    self.disabled_up = False
            else:
                self.disabled_alpha = max(self.disabled_alpha - 8 * self.main.ratio_fps, 0)
                self.disabled_text.set_alpha(self.disabled_alpha)

                if self.disabled_alpha == 0:
                    self.disabled_up = True
                    self.disabled_showing = False

            self.main.screen.blit(self.disabled_text, self.disabled_rect)
        """""""""

    # fonction de préparation d'une nouvelle partie
    def reset_board(self, newgame=False):
        self.board = np.zeros((8, 8)) # reset du plateau

        # position de départ
        self.board[3, 3:5] = [1, -1]
        self.board[4, 3:5] = [-1, 1]

        self.player_turn = -1 # choix du joueur qui commence
        self.ddqn_number *= -1 # couleur de l'algorithme dqn
        self.returned_pieces = {} # reset du dictionnaire du dernier coup joué

        if self.main.mod != 3:
            self.board_shown = self.board.copy()
            self.board_shown_number = 0

            # remise à 0 des chronos
            self.timer_last_time = pygame.time.get_ticks()
            self.timer_player_1 = self.settings_timer_list[self.main.memory.datas["timer"]] * 60000
            self.timer_player_2 = self.settings_timer_list[self.main.memory.datas["timer"]] * 60000
            self.timer_player_1_text = self.timer_text_font.render(f"{self.timer_player_1 // 60000}:{self.timer_player_1 % 60000 // 1000:02}", 1,(255, 255, 255) if self.player_1 == -1 else (0, 0, 0))
            self.timer_player_2_text = self.timer_text_font.render(f"{self.timer_player_2 // 60000}:{self.timer_player_2 % 60000 // 1000:02}", 1,(255, 255, 255) if self.player_1 == 1 else (0, 0, 0))
            self.timer_player_1_text_rect = self.timer_player_1_text.get_rect(topright=(self.game_board_rect.bottomright[0] - 30,self.game_board_rect.bottomright[1] + 12))
            self.timer_player_2_text_rect = self.timer_player_2_text.get_rect(bottomright=(self.game_board_rect.topright[0] - 30,self.game_board_rect.topright[1] - 10))
            self.timer_player_1_back = pygame.Rect(0, 0,self.timer_player_1_text.get_width() + 10, self.timer_player_1_text.get_height() + 5)
            self.timer_player_1_back.center = self.timer_player_1_text_rect.center
            self.timer_player_2_back = pygame.Rect(0, 0,self.timer_player_1_text.get_width() + 10, self.timer_player_1_text.get_height() + 5)
            self.timer_player_2_back.center = self.timer_player_2_text_rect.center

            if not newgame:
                # retour aux profils prédéfinies
                self.profil_player_1_picture = self.profil_pictures["original"][1]
                self.profil_player_2_picture = self.profil_pictures["original"][-1]

                self.profil_player_1_text = self.profils_text["player_1"]
                self.profil_player_2_text = self.profils_text["player_2"]
            elif self.main.mod == 1:
                self.player_1 = choice([-1, 1])
                self.profil_player_1_picture = self.profil_pictures["original"][self.player_1]
                self.profil_player_2_picture = self.profil_pictures["original"][-self.player_1]
                self.not_ended = True
            elif self.main.mod == 2:
                self.player_1 = choice([-1, 1])
                self.profil_player_1_picture = self.profil_pictures["original"][self.player_1]
                self.profil_player_2_picture = self.profil_pictures[self.bot][-self.player_1]
                self.not_ended = True

            # remise à False des différentes variables de vérification
            self.cant_play = False
            self.is_prepared = False

            # reset du stockage de la partie
            self.transitions = []
            self.transitions.append((self.board.copy(), self.player_turn, 1, 1))
            self.moves = []
            self.rewards = []

            current_move = f"{len(self.moves)}.    -"

            current_move_text = self.moves_font.render(current_move, 1, (255, 255, 255))
            current_move_text_rect = current_move_text.get_rect(midleft=(
                self.moves_initial_x + self.moves_x_space * (len(self.moves) % 2),
                self.moves_initial_y + self.moves_y_space * (len(self.moves) // 2)
            ))
            current_move_back = pygame.Rect(0, 0, current_move_text.get_width() + 3, current_move_text.get_height() + 3)
            current_move_back.center = current_move_text_rect.center
            self.moves.append((current_move_back, current_move_text, current_move_text_rect))

        # reset des scores
        self.white_score = np.sum(self.board == 1)
        self.black_score = np.sum(self.board == -1)

    # fonction de détermination du tour
    def turn(self):
        mixed_board, return_board = self.check_valid_play(self.board, self.player_turn)

        # détermination du joueur du tour
        if self.main.mod == 1:
            self.waiting_for = "human"

        elif self.main.mod == 2 and self.bot == "minmax":
            self.waiting_for = "human" if self.player_turn == self.player_1 else "minmax"

        elif self.main.mod == 2 and self.bot == "ddqn":
            self.waiting_for = "human" if self.player_turn == self.player_1 else "ddqn"

        if 2 in mixed_board:  # vérifie qu'on coup au moins est jouable
            self.cant_play = False
            self.is_prepared = True

        elif self.cant_play:  # fin de partie
            # aucun des deux joueurs ne peut jouer
            current_move = f"{len(self.moves)}.   -"
            current_move_text = self.moves_font.render(current_move, 1, (255, 255, 255))
            current_move_text_rect = current_move_text.get_rect(midleft=(
                self.moves_initial_x + self.moves_x_space * (len(self.moves) % 2),
                self.moves_initial_y + self.moves_y_space * (len(self.moves) // 2)
            ))
            current_move_back = pygame.Rect(0, 0, current_move_text.get_width() + 3, current_move_text.get_height() + 3)
            current_move_back.center = current_move_text_rect.center
            self.moves.append((current_move_back, current_move_text, current_move_text_rect))
            self.transitions.append((self.board.copy(), self.player_turn, self.last_move_rect, self.returned_pieces))
            self.board_shown_number = len(self.moves) - 1

            self.do_results()

        else:
            # {self.player_turn} ne peut pas jouer, il passe donc son tour.")
            self.cant_play = True
            self.player_turn *= -1
            self.board_shown_number = len(self.moves) - 1

            current_move = f"{len(self.moves)}.   -"
            current_move_text = self.moves_font.render(current_move, 1, (255, 255, 255))
            current_move_text_rect = current_move_text.get_rect(midleft=(
                self.moves_initial_x + self.moves_x_space * (len(self.moves) % 2),
                self.moves_initial_y + self.moves_y_space * (len(self.moves) // 2)
            ))
            current_move_back = pygame.Rect(0, 0, current_move_text.get_width() + 3, current_move_text.get_height() + 3)
            current_move_back.center = current_move_text_rect.center
            self.moves.append((current_move_back, current_move_text, current_move_text_rect))
            self.transitions.append((self.board.copy(), self.player_turn, self.last_move_rect, self.returned_pieces))

            if self.waiting_for == "human":
                self.preparation_cooldown = pygame.time.get_ticks()

        return mixed_board, return_board

    # fonction du tour du joueur humain
    def human_turn(self, x, y):
        if self.mixed_board[x, y] == 2:  # vérification que le coup est valide
            self.unplayable_cnt = 4
            self.board[x, y] = self.player_turn  # placement du pion joué
            self.do_return_pieces(x, y, self.board, self.return_board, self.player_turn)  # retournement des autres pions
            self.next_player(x, y)
            self.preparation_cooldown = pygame.time.get_ticks()
            self.sound_manager.play("place")
            self.timer_last_time = pygame.time.get_ticks()

        elif self.game_board_x + 50 + x * 100 != self.unplayable_x or self.game_board_y + 50 + y * 100 != self.unplayable_y or self.unplayable_cnt >= 3:
            self.unplayable_x = self.game_board_x + 50 + y * 100 # x pour les colonnes
            self.unplayable_y = self.game_board_y + 50 + x * 100 # y pour les lignes
            self.unplayable_rect = self.unplayable.get_rect(center=(self.unplayable_x, self.unplayable_y))
            self.unplayable_draw = True
            self.unplayable_cooldown = pygame.time.get_ticks()
            self.unplayable_cnt = 1
            self.sound_manager.play("unplayable_move")

    # tour de l'agent DDQN
    def ddqn_turn(self, mixed_board, return_board):
        # choix du coup par l'agent
        move = self.main.ddqn.main_network.select_action(mixed_board, self.board, self.player_turn, self.main.ddqn.epsilon if self.main.mod == 3 else 0)
        x, y = move

        # sauvegarde du plateau
        current_board = self.board.copy()

        # application du coup
        self.board[x, y] = self.player_turn
        self.do_return_pieces(x, y, self.board, return_board, self.player_turn)

        # passage au joueur suivant
        self.player_turn *= -1

        # vérification que la partie n'est pas fini
        next_mixed_board, next_return_board = self.check_valid_play(self.board, self.player_turn)  # vérification des coups valides
        self.main.ddqn.done = True if np.sum(next_mixed_board == 2) == 0 else False

        if self.main.ddqn.done:
            self.player_turn *= -1
            end_mixed_board, _ = self.check_valid_play(self.board, self.player_turn)
            self.main.ddqn.done = True if np.sum(end_mixed_board == 2) == 0 else False
        else:
            self.player_turn *= -1

        # calcul de la récompense en fonction du coup
        reward = self.main.ddqn.get_reward(current_board, self.board, mixed_board, next_mixed_board)

        if self.main.mod == 2:
            print(f"DDQN ({self.player_turn}) joue {move} avec un récompense de {reward}")
            self.next_player(x, y)
        else:
            # mise à jour du modèle DQN avec les transitions mémorisées
            self.main.ddqn.learn()
            self.main.ddqn.temp_memory.append([current_board, (x, y), reward, self.board.copy(), self.main.ddqn.done, self.player_turn, self.mixed_board, next_mixed_board])
            self.player_turn *= -1
        self.sound_manager.play('place') if self.main.mod == 2 else None

    # tour de l'agent Minmax
    def minmax_turn(self, return_board):
        # choix du coup par l'agent Minmax
        move = self.main.minmax.find_best_move(self.board.copy())
        x, y = move

        # application du coup
        self.board[x, y] = self.player_turn
        self.do_return_pieces(x, y, self.board, return_board, self.player_turn)

        if self.main.mod == 2:
            print(f"MinMax ({self.player_turn}) joue {move}")
            self.next_player(x, y)
        else:
            self.player_turn *= -1
        self.sound_manager.play('place') if self.main. mod == 2 else None

    # tour de l'ancienne version de DQN
    def old_dqn_turn(self, mixed_board, return_board):
        # choix du coup par l'agent
        move = self.main.old_dqn.main_network.select_action(mixed_board, self.board, self.player_turn, self.main.old_dqn.epsilon)
        # valid_moves = np.argwhere(mixed_board == 2)
        # move = choice(valid_moves)
        x, y = move

        # application du coup
        self.board[x, y] = self.player_turn
        self.do_return_pieces(x, y, self.board, return_board, self.player_turn)

        if self.main.mod == 2:
            print(f"Old_DQN ({self.player_turn}) joue {move}")

        self.player_turn *= -1

    # passage au joueur suivant
    def next_player(self, x, y):
        self.player_turn *= -1
        self.is_prepared = False

        if len(self.moves) < 10:
            current_move = f"{len(self.moves)}.   {'abcdefgh'[x]}{8 - y}"
        else:
            current_move = f"{len(self.moves)}. {'abcdefgh'[x]}{8 - y}"

        current_move_text = self.moves_font.render(current_move, 1, (255, 255, 255))
        current_move_text_rect = current_move_text.get_rect(midleft=(
            self.moves_initial_x + self.moves_x_space * (len(self.moves) % 2),
            self.moves_initial_y + self.moves_y_space * (len(self.moves) // 2)
        ))
        current_move_back = pygame.Rect(0, 0, current_move_text.get_width() + 3, current_move_text.get_height() + 3)
        current_move_back.center = current_move_text_rect.center
        self.moves.append((current_move_back, current_move_text, current_move_text_rect))

        self.board_shown = self.board.copy()
        self.board_shown_number = len(self.moves) - 1

    # fonction vérifiant les coups valides
    def check_valid_play(self, board, player_turn):
        self.valid_moves_mask = np.zeros_like(board, dtype=bool)  # masque pour les cases adjacentes valides
        return_board = np.full(board.shape, "", dtype=object)  # tableau pour accumuler les directions

        rows, cols = board.shape  # obtention des dimensions du plateau (lignes, colonnes)

        for cnt, (dx, dy) in enumerate(self.directions):  # parcours des directions

            for x in range(rows):  # boucle sur chaque ligne
                for y in range(cols):  # boucle sur chaque colonne

                    if board[x, y] == 0:  # vérifie si la case est vide
                        enemy_found = False  # indicateur pour savoir si un pion ennemi a été trouvé

                        for step in range(1, 8):  # vérifie jusqu'à 7 cases dans la direction
                            nx, ny = x + step * dx, y + step * dy  # calcule la position suivante

                            if not (0 <= nx < rows and 0 <= ny < cols):  # vérifie les limites du plateau
                                break  # break si on sort du plateau

                            if board[nx, ny] == -player_turn:  # vérifie si c'est un pion ennemi
                                enemy_found = True  # un pion ennemi a été trouvé

                            elif board[nx, ny] == player_turn and enemy_found:  # vérifie un pion allié après un ennemi
                                self.valid_moves_mask[x, y] = True  # marque la case comme valide
                                return_board[x, y] += str(cnt)  # ajoute la direction à return_board
                                break  # break de la boucle si un pion allié est trouvé

                            else:
                                break  # break si on trouve une case vide ou un pion allié sans pion ennemi

        mixed_board = np.copy(board)  # initialisation de mixed_board
        mixed_board[self.valid_moves_mask] = 2  # remplacement des coups valides par des 2
        return mixed_board, return_board  # renvoie le plateau avec les coups jouables et les directions à capturer

    # fonction retournant les pions
    def do_return_pieces(self, x_choice, y_choice, board, return_board, player_turn, treesearch=False):
        self.returned_pieces = {}
        for i in range(len(return_board[x_choice][y_choice])):
            stop = False
            counter = 0

            while not stop:
                counter += 1

                if board[x_choice + counter * self.directions[int(return_board[x_choice][y_choice][i])][0]][y_choice \
                    + counter * self.directions[int(return_board[x_choice][y_choice][i])][1]] == -player_turn:

                    board[x_choice + counter * self.directions[int(return_board[x_choice][y_choice][i])][0]][y_choice \
                        + counter * self.directions[int(return_board[x_choice][y_choice][i])][1]] = player_turn

                    self.returned_pieces[((x_choice + counter * self.directions[int(return_board[x_choice][y_choice][i])][0],
                        y_choice + counter * self.directions[int(return_board[x_choice][y_choice][i])][1]))] = None
                else:
                    stop = True

                if counter == 7:
                    stop = True

        if self.main.mod != 3 and not treesearch:
            # sauvegarde des dernières pièces retournées
            self.last_move_x = self.game_board_x + 50 + y_choice * 100 # x pour les colonnes
            self.last_move_y = self.game_board_y + 50 + x_choice * 100 # y pour les lignes
            self.last_move_rect = self.last_move.get_rect(center=(self.last_move_x, self.last_move_y))

            for piece in self.returned_pieces.keys():
                last_return = pygame.Surface((100, 100))
                last_return.fill(self.game_board_dict[self.main.memory.datas["board"]][2])
                last_return.set_alpha(180)
                last_return_x = self.game_board_x + 50 + piece[1] * 100
                last_return_y = self.game_board_y + 50 + piece[0] * 100
                last_return_rect = last_return.get_rect(center=(last_return_x, last_return_y))
                self.returned_pieces[piece] = (last_return, last_return_rect)

            self.transitions.append((self.board.copy(), -self.player_turn, self.last_move_rect, self.returned_pieces))

        return board

    # fonction de fin de partie
    def do_results(self, resign=False):
        self.mixed_board[self.mixed_board == 2] = 0
        self.black_score = np.sum(self.board == -1)  # calcul du score noir
        self.white_score = np.sum(self.board == 1)  # calcul du score blanc

        delta = self.black_score - self.white_score  # calcul de la différence des deux scores
        self.main.state = "showing_results"
        self.results_player_1_picture = pygame.transform.scale(self.profil_player_1_picture, (70, 70))
        self.results_player_2_picture = pygame.transform.scale(self.profil_player_2_picture, (70, 70))

        # détermination du gagnant
        if resign:
            winner = -self.player_turn
        elif delta != 0:
            winner = -1 if delta > 0 else 1
        else:
            winner = 0

        if winner != 0:  # il y a un gagnant et un perdant
            if winner == -1:
                self.results_winner_text = self.results_winner_font.render(f"Les noirs ont gagné", 1, (255, 255, 255))
            else:
                self.results_winner_text = self.results_winner_font.render(f"Les blancs ont gagné", 1, (255, 255, 255))
            self.results_winner_text_rect = self.results_winner_text.get_rect(center=(self.results_back.midtop[0], self.results_back.midtop[1] + self.results_winner_text.get_height()))

            self.results_player_1_score = self.scores_bar_font.render(f"{self.white_score if self.player_1 == 1 else self.black_score}", 1, (255, 255, 255))
            self.results_player_2_score = self.scores_bar_font.render(f"{self.black_score if self.player_1 == 1 else self.white_score}", 1, (255, 255, 255))
            self.results_player_1_score_rect = self.results_player_1_score.get_rect(midtop=self.results_player_1_picture_rect.midbottom)
            self.results_player_2_score_rect = self.results_player_2_score.get_rect(midtop=self.results_player_2_picture_rect.midbottom)

            self.not_ended = False

        else:  # il y a égalité
            self.results_winner_text = self.results_winner_font.render(f"Pas de gagnant", 1, (255, 255, 255))
            self.results_winner_text_rect = self.results_winner_text.get_rect(center=(self.results_back.midtop[0],self.results_back.midtop[1] + self.results_winner_text.get_height()))

            self.results_player_1_score = self.scores_bar_font.render(f"{self.white_score if self.player_1 == 1 else self.black_score}", 1, (255, 255, 255))
            self.results_player_2_score = self.scores_bar_font.render(f"{self.black_score if self.player_1 == 1 else self.white_score}", 1, (255, 255, 255))
            self.results_player_1_score_rect = self.results_player_1_score.get_rect(midtop=self.results_player_1_picture_rect.midbottom)
            self.results_player_2_score_rect = self.results_player_2_score.get_rect(midtop=self.results_player_2_picture_rect.midbottom)

            self.not_ended = False

        pygame.mixer.music.set_volume(self.main.memory.datas["volume"] / 800)
        self.sound_manager.play("winning")