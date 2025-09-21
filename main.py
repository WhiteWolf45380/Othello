from game import Game
from pieces import FakePiece
import numpy as np
import pygame
from math import sqrt
import json
from collections import deque
import sys
import os

'''''''''

Packages nécessaires au bon fonctionnement du programme :

    Pour le jeu en lui_même :
        ► numpy
        ► pygame
    
    Pour les bots :
        ► torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ► matplotlib

Transformation en éxécutable:
    ► pyinstaller --onefile --windowed --icon="assets/logo.ico" --add-data "assets;assets" --add-data "export;export" main.py

'''''''''

# class stockant les paramètres à conserver entre les éxécutions
class Memory:
    def __init__(self, main):
        self.filename = main.get_path("export/datas.json") # path de stockage des paramètres
        self.datas = {  # paramètres fixés à une valeur par défaut
            "fps_max": 30,  # la limite de fps
            "volume": 50,  # le volume sonore
            "timer": 0,  # le timer des joueurs durant une partie
            "board": "wood",  # style du plateau
        }
        self.load() # chargement des paramètres

    # fonction de sauvegarde
    def save(self):
        try:
            with open(self.filename, "w") as exported_datas:  # ouverture du fichier
                json.dump(self.datas, exported_datas, indent=4)  # exportation des paramètres

        except json.JSONDecodeError as e:  # en cas d'erreur lors de la sauvegarde
            print("Erreur de sauvegarde dans le JSON", e)  # message d'erreur dans la console

    # fonction de chargement
    def load(self):
        try:
            with open(self.filename, "r") as imported_datas:  # ouverture du fichier
                self.datas = json.load(imported_datas)  # importation des paramètres

        except json.JSONDecodeError as e:  # en cas d'erreur lors du chargement
            print("Erreur de chargement dans le JSON", e)  # message d'erreur dans la console


# class principale du jeu
class Othello:
    def __init__(self):

        self.memory = Memory(self)# récupération des données sauvegardées

        # initialisation à None de différents modèles de bots
        self.ddqn = None
        self.minmax = None
        self.old_dqn = None

        self.mod = 0  # choix des adversaires
        self.n = 0  # nombre de parties si deux bots s'affrontent
        self.winner = 0  # gagnant de la partie
        self.winner_score = 0  # score du gagnant
        self.loser_score = 0  # score du perdant

        self.running = True  # boucle du jeu
        self.screen_width = 1920  # largeur initiale de l'écran
        self.screen_height = 1080  # hauteur initiale de l'écran
        self.screen = pygame.Surface((1920, 1080))  # écran du jeu
        self.fullscreen = False
        self.screen_resized = pygame.display.set_mode((1280, 720), pygame.RESIZABLE) # écran affiché
        self.clock = pygame.time.Clock() # initialisation de clock pour la gestion des fps
        self.frozen_screen = None  # variable utilisée pour sauvegarder l'écran à un moment donné lors de l'ouverture des paramètres
        self.fps = deque(maxlen=5)  # initialisation d'un buffer pour éviter les grosses variations lors du calcul des fps
        for _ in range(5):
            self.fps.append(self.memory.datas["fps_max"])  # ajout par défaut des fps limites
        self.fps_timer = 0  # timer suivant la durée que prend un cycle complet à s'éxécuter
        self.music_cooldown = 0

        # pour calculer la vitesse des animations, on utilise un ratio entre les fps actuels et 30 fps. Par défaut, on prend les fps limites comme fps actuels.
        self.ratio_fps = 30 / self.memory.datas["fps_max"]

        pygame.init()  # initialisation de pygame

        self.game = Game(self)  # création de l'environnement de jeu Othello
        self.game.sound_manager.music("game_loading")
        self.state = "starting" # indique l'état actuel de l'écran, permettant la gestion des éléments à afficher
        self.state_save = self.state # permet de sauvegarder l'état de jeu avant ouverture des paramètres
        self.step = 1  # progression dans l'animation de lancement du jeu

        # écran de chargement des bots
        self.loading_font = pygame.font.Font(self.get_path("assets/fonts/loading.ttf"), 100)  # initialisation de la police
        self.loading_text = self.loading_font.render("Préparation du Modèle...", 1, (255, 0, 0))  # création du texte
        self.loading_text_back = self.loading_font.render("Préparation du Modèle...", 1, (0, 0, 0)) # ombre du texte
        self.loading_text_rect = self.loading_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))  # rect du texte
        self.loading_text_back_rect = self.loading_text.get_rect(center=(self.screen_width / 2 + 7, self.screen_height / 2 + 7))  # rect de l'ombre du texte

        self.loading_under_font = pygame.font.Font(self.get_path("assets/fonts/loading.ttf"), 25)  # initialisation de la police
        self.loading_under_text = self.loading_under_font.render("(peut prendre plusieurs minutes au premier chargement)", 1, (255, 0, 0))  # création du texte
        self.loading_under_text_back = self.loading_under_font.render("(peut prendre plusieurs minutes au premier chargement)", 1, (0, 0, 0)) # ombre du texte
        self.loading_under_text_rect = self.loading_under_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 80))  # rect du texte
        self.loading_under_text_back_rect = self.loading_under_text.get_rect(center=(self.screen_width / 2 + 7, self.screen_height / 2 + 87))  # rect de l'ombre du texte

    @staticmethod # fonction returnant les fps, qui sont par défaut mis à 30
    def fps_method(fps=30):
        return fps  # renvoie la valeur des fps

    @staticmethod
    def get_path(relative_path): # fonction permettant d'adapté les path  àl'éxécutable
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)

    # fonction de lancement du jeu
    def start_othello(self):
        icon = pygame.image.load(self.get_path('assets/game_logo.jpg'))  # icone de la page du jeu
        pygame.display.set_caption('Othello')  # titre de la page du jeu
        pygame.display.set_icon(icon)  # application de l'icone

        # assets pygame de l'animation du lancement du jeu

        # ombre entourant le plateau
        shadow = pygame.image.load(self.get_path('assets/shadow.png')) # chargement de l'image
        shadow = pygame.transform.scale(shadow, (self.screen_width, self.screen_height))  # redimensionnement de l'image

        # titre du jeu
        title_font = pygame.font.Font(self.get_path("assets/fonts/title.ttf"), 200)  # initialisation de la police
        title_text = title_font.render("OTHELLO", 1, (255, 255, 255))  # création du texte
        title_border = title_font.render("OTHELLO", 1, (0, 0, 0))  # création de l'ombre
        title_text_rect = title_text.get_rect(midtop=(self.screen_width / 2, self.screen_height / 2 - 440)) # placement du titre
        title_alpha = 0  # opacité par défaut
        title_cooldown = 0  # cooldown avant d'incrémenter l'opacité

        # pièces du plateau
        all_fake_pieces = pygame.sprite.Group()  # création du groupe contenant les pièces utiles à l'animation de chargement du jeu
        for i in range(4):  # 4 pièces pour le placement intial lors d'une partie d'othello
            fake_piece = FakePiece(self, i)  # création d'un sprite FakePiece
            all_fake_pieces.add(fake_piece)  # ajout du sprite au groupe de sprites

        # boutton start
        start_button = pygame.image.load(self.get_path("assets/start_button_0.xcf"))  # chargement de l'image
        start_button_original = start_button.copy()  # création d'une copie pour conserver l'orientation initiale
        start_button_x = self.screen_width / 2  # placement x du boutton
        start_button_y = -50  # placement y par défaut du boutton
        start_button_y_final = self.screen_height - 240  # placement y final du boutton
        start_button_rect = start_button.get_rect(center=(start_button_x, start_button_y))  # rect du boutton
        start_button_velocity_rate = 1.0  # ratio de la vélocité, créant un ralentissement lorsque le boutton approche de sa position finale
        start_button_angle = 0  # angle du boutton
        start_button_cooldown = 0  # cooldown avant déplacement et rotation du boutton

            # flash
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.fill((255, 255, 255))
        overlay_alpha = 100
        overlay.set_alpha(overlay_alpha)

        try:
            while self.running:  # boucle du jeu
                self.fps.append(round(min(max(1000 // (pygame.time.get_ticks() - self.fps_timer), 10), self.memory.datas["fps_max"]), 2))  # ajout des fps réels au buffer
                self.fps_timer = pygame.time.get_ticks()  # reset du timer
                self.ratio_fps = round(30 / (sum(self.fps) // len(self.fps)), 2)  # calcul de la moyenne des fps du buffer pour aligner la vitesse des animations aux limites de l'appareil

                # fondus de la musique
                if self.game.sound_manager.special_channel.get_busy():
                    pygame.mixer.music.set_volume(self.memory.datas["volume"] / 600)
                elif pygame.mixer.music.get_volume() < self.memory.datas["volume"] / 200 and pygame.time.get_ticks() - self.music_cooldown > 50:
                    pygame.mixer.music.set_volume(min(pygame.mixer.music.get_volume() + max((self.memory.datas["volume"] / 200 - pygame.mixer.music.get_volume()) / 20, 0.01), self.memory.datas["volume"] / 200))
                    self.music_cooldown = pygame.time.get_ticks()

                self.screen.fill((0, 0, 0)) if self.state == "starting" else self.screen.blit(self.game.background, self.game.background_rect)  # arrrière plan

                if self.state == "starting":  # écran de lancement du jeu
                    if 6 > self.step >= 1:  # première étape de l'animation du lancement

                        if self.game.game_board_scale < 1.0 or self.game.game_board_overlay_alpha > 0 >= self.game.game_board_cooldown:  # effet de zoom avec fondu sur le plateau
                            # zoom
                            self.game.game_board_scale = min(1, self.game.game_board_scale + round(0.005 * self.ratio_fps, 3))  # limite la taille maximale

                            # calcul des dimensions entières pour éviter les imprécisions
                            scaled_width = int(self.game.game_board_width * self.game.game_board_scale)
                            scaled_width += 1 if scaled_width % 2 != 0 else 0
                            scaled_height = int(self.game.game_board_height * self.game.game_board_scale)
                            scaled_height += 1 if scaled_height % 2 != 0 else 0

                            self.game.game_board = pygame.transform.scale(self.game.game_board_original,(scaled_width, scaled_height))

                            # fondu
                            self.game.game_board_overlay_alpha = max(0, self.game.game_board_overlay_alpha - min(round((170 / max(self.game.game_board_overlay_alpha, 1))**2 * self.ratio_fps, 1), 3))  # réduction de l'opacité du voile
                            self.game.game_board_overlay.set_alpha(self.game.game_board_overlay_alpha)  # application de l'opacité

                            self.game.game_board_rect = self.game.game_board.get_rect(center=(self.screen_width // 2, self.screen_height // 2))  # recentrage du plateau
                        elif self.step == 1:
                            self.step += 1

                        self.screen.blit(self.game.game_board, self.game.game_board_rect)  # affichage du plateau
                        self.screen.blit(self.game.game_board_overlay, (0, 0))  # affichage du voile

                    self.screen.blit(shadow, (0, 0))  # affichage de l'ombre entourant le plateau

                    if 6 > self.step >= 2:
                        if self.step < 3:
                            current_piece = list(all_fake_pieces)[int(self.step % 2 * 10)] # gestion des pièces une par une
                            current_piece.velocity_rate *= current_piece.velocity_decayrate if abs(current_piece.y_placement - current_piece.y_placement_max) < 100 else 1 # diminution de la vitesse lorsque la piece approche sa position finale

                            current_piece.hidden = False  # affichage de la pièce actuelle
                            current_piece.y_placement = min(current_piece.y_placement + current_piece.velocity * current_piece.velocity_rate,current_piece.y_placement_max)  # déplacement de la pièce en fonction de sa vélocité

                            for fake_piece in all_fake_pieces:
                                fake_piece.placement() # fonction d'actualisation des pièces

                            if current_piece.y_placement == current_piece.y_placement_max:
                                self.step += 0.1 # passage à la pièce suivante
                                self.game.sound_manager.play("place")

                            if self.step >= 2.4:  # si l'on atteint la dernière pièce, passage à la prochaine étape
                                self.step = 3
                                title_cooldown = pygame.time.get_ticks()  # coolodown du fondu du titre
                        else:
                            all_fake_pieces.draw(self.screen)  # affichage des pièces sur l'écran

                    if 6 > self.step >= 3 and pygame.time.get_ticks() - 500 >= title_cooldown:
                        if title_alpha < 255:  # fondu si l'opacité n'est pas encore à son maximum
                            title_alpha = min(255, title_alpha + round(2 * self.ratio_fps, 1))  # réduction de l'opacité du voile
                            title_text.set_alpha(title_alpha)  # application de alpha
                            title_border.set_alpha(title_alpha)  # application de alpha à l'ombre
                        elif self.step == 3:  # si le fondu est finit, passage à l'étape suivante
                            self.step += 1

                        self.screen.blit(title_border, (title_text_rect.left + 5, title_text_rect.top + 3))  # affichage de l'ombre du titre
                        self.screen.blit(title_text, title_text_rect)  # affichage du titre

                    if 6 > self.step >= 4:
                        if start_button_y < start_button_y_final or start_button_angle != 0:  # déplacement et rotation du bouton jusqua sa position finale
                            # chute du boutton
                            start_button_velocity = round(11 * self.ratio_fps)  # actualisation de la vélocité en fonction des fps
                            start_button_velocity_rate *= 1 - round(0.09 * self.ratio_fps, 2) if abs(start_button_y_final - start_button_y) <= 50 else 1  # actualisation de la vélocité en fonction de la position
                            start_button_y = min(start_button_y_final, start_button_y + start_button_velocity * start_button_velocity_rate)  # déplacement du boutton

                            # rotation du boutton
                            if abs(start_button_y_final - start_button_y) <= 30 and (start_button_angle % 360 >= 360 - round(13 * self.ratio_fps) or start_button_angle == 0):
                                start_button_angle = 0  # maintient de l'angle  à 0 une fois la position finale atteinte
                            elif abs(start_button_y_final - start_button_y) <= 30:
                                start_button_angle = min(start_button_angle + round(13 * self.ratio_fps), 360 * start_button_angle // 360 + (360 - (start_button_angle % 360)))  # rotation en fonction de l'angle restant
                            else:
                                start_button_angle += round(13 * self.ratio_fps)  # rotation normale du boutton
                            start_button = pygame.transform.rotate(start_button_original, start_button_angle)  # application de la rotation

                            # mise à jour du centre
                            start_button_rect = start_button.get_rect(center=(start_button_x, start_button_y))
                        elif self.step == 4:
                            self.step += 1  # passage à la dernière étape
                            start_button_cooldown = pygame.time.get_ticks()  # coolown du clignottement du boutton

                        self.screen.blit(start_button, start_button_rect)  # affichage du boutton

                    if self.step == 5:
                        if pygame.time.get_ticks() - 400 >= start_button_cooldown:  # clignottement
                            start_button = pygame.image.load(self.get_path("assets/start_button_1.xcf")) if start_button == start_button_original else start_button_original # alterne entre les deux images
                            start_button_cooldown = pygame.time.get_ticks()  # cooldown du clignottement

                elif 6 <= self.step <= 7:
                    self.game.game_update()

                    if self.step == 6:
                        overlay_alpha = min(180, overlay_alpha + 25 * self.ratio_fps)
                        overlay.set_alpha(overlay_alpha)
                        if overlay_alpha == 180:
                            self.step = 7

                    elif self.step == 7:
                        overlay_alpha = max(0, overlay_alpha - 7 * self.ratio_fps)
                        overlay.set_alpha(overlay_alpha)
                        self.screen.blit(overlay, (0, 0))
                        if overlay_alpha == 0:
                            self.step = 0

                    self.screen.blit(overlay, (0, 0))

                else:
                    self.game.game_update()  # actualisation de l'écran

                x, y = pygame.mouse.get_pos()  # récupération des coordonées de la souris
                self.game.mouse_x = x / (self.screen_resized.get_width() / 1920)
                self.game.mouse_y = y / (self.screen_resized.get_height() / 1080)

                # capture des touches
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # vérifie que la fenêtre n'est pas fermée
                        pygame.mixer.music.stop()
                        self.running = False

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # si un bouton de la souris est pressé
                        event.pos = (event.pos[0] / (self.screen_resized.get_width() / 1920),
                                     event.pos[1] / (self.screen_resized.get_height() / 1080))
                        if start_button_rect.collidepoint(event.pos) and self.step == 5: # clique sur le boutton START
                            self.step = 6
                            self.state = "choosing_mod"  # passage de l'état au choix du mode
                            self.game.game_board_overlay.set_alpha(235)  # le voile est réutilisé pour un autre écran, d'où sa mise à une opacité de 235
                            self.game.game_board = self.game.game_board_dict[self.memory.datas["board"]][0]  # style du plateau de jeu
                            self.game.last_move.fill(self.game.game_board_dict[self.memory.datas["board"]][1])  # couleur du surlignage en fonction du style
                            self.game.sound_manager.play("game_start")
                            self.game.sound_manager.music("music_loop")

                        elif self.state == "choosing_mod" and self.game.mods_background[0].collidepoint(event.pos): # choisit le jeu en local
                            self.mod = 1  # le mode 1 correspond au JcJ
                            self.game.not_ended = True  # démarrage de la partie
                            self.state = "playing"  # passage de l'état à "en jeu"
                            self.game.timer_last_time = pygame.time.get_ticks()  # démarrage des timers
                            self.game.reset_board(newgame=True)  # préparation du plateau
                            self.game.sound_manager.play("mod_choosen")

                        elif self.state == "choosing_mod" and self.game.mods_background[1].collidepoint(event.pos): # choisit le jeu contre des robots
                            self.state = "choosing_bot"  # passage de l'état au choix du robot
                            self.game.not_ended = True  # démarage de la partie
                            self.mod = 2  # le mode 2 correspond au JcE
                            self.game.sound_manager.play("mod_choosen")

                        elif self.state == "choosing_mod" and self.game.mods_background[2].collidepoint(event.pos): # choisit l'entrainement des robots
                            self.state = "choosing_bot"  # passage de l'état au choix du robot
                            self.mod = 3  # le mode 3 correspond à l'entrainement des robots
                            self.game.sound_manager.play("mod_choosen")
                            # self.game.disabled_showing = True

                        elif self.state == "choosing_bot" and self.game.bot_minmax_back.collidepoint(event.pos) and self.mod == 2:  # choisit le robot
                            self.game.bot = "minmax"  # application du choix
                            self.game.not_ended = True  # démarrage de la partie
                            self.game.sound_manager.play("mod_choosen")

                            # création d'une instance de Minmax si elle n'existe pas déja
                            if self.minmax is None:
                                # écran de chargement du bot
                                self.screen.blit(self.game.game_board_overlay, (0, 0))
                                self.screen.blit(self.loading_text_back, self.loading_text_back_rect)
                                self.screen.blit(self.loading_text, self.loading_text_rect)
                                self.screen.blit(self.loading_under_text_back, self.loading_under_text_back_rect)
                                self.screen.blit(self.loading_under_text, self.loading_under_text_rect)
                                self.screen_resized.blit(pygame.transform.scale(self.screen, (self.screen_resized.get_width(), self.screen_resized.get_height())), (0, 0))
                                pygame.display.update()

                                from Minmax_file import MinMax  # importation de Minmax
                                self.minmax = MinMax(self, depth=3)  # création d'une instance de Minmax

                            if self.mod == 2:  # si JcE
                                self.state = "playing" # passage de l'état à "en jeu"
                                self.game.reset_board(newgame=True) # préparation du plateau

                            else: # si entrainement
                                self.init_training() # démarrage de l'entrainement

                        elif self.state == "choosing_bot" and self.game.bot_ddqn_back.collidepoint(event.pos):  # choix du bot DDQN
                            self.game.bot = "ddqn"  # application du choix
                            self.game.not_ended = True # démarrage de la partie

                            # création de deux instances de DQN, pour une meilleure stabilité lors de l'entrainement
                            if self.ddqn is None:
                                # écran de chargement des bots
                                self.screen.blit(self.game.game_board_overlay, (0, 0))
                                self.screen.blit(self.loading_text_back, self.loading_text_back_rect)
                                self.screen.blit(self.loading_text, self.loading_text_rect)
                                self.screen.blit(self.loading_under_text_back, self.loading_under_text_back_rect)
                                self.screen.blit(self.loading_under_text, self.loading_under_text_rect)
                                self.screen_resized.blit(pygame.transform.scale(self.screen, (self.screen_resized.get_width(), self.screen_resized.get_height())), (0, 0))
                                pygame.display.update()

                                from DQN_file import DDQN  # importation de l'algorithme DQN
                                self.ddqn = DDQN(self, double=True)
                                self.ddqn.update_target_network()
                                self.old_dqn = DDQN(self) # paramètres actualisés toutes les 2500 parties

                            if self.mod == 2:  # si JcE
                                self.state = "playing"  # passage de l'état à "en jeu"
                                self.game.reset_board(newgame=True) # préparation du plateau
                            else:  # si entrainement
                                self.init_training()

                        elif self.state != "starting" and self.game.sidebar_cursor_rect.collidepoint(event.pos): # si le joueur clique sur le volet latéral
                            if self.game.sidebar_cursor_current == "log out": # sur le boutton quitter
                                pygame.mixer.music.stop()
                                pygame.display.quit()
                                print("Fermeture du jeu...")
                                self.game.sound_manager.play("mouse_click")
                                self.memory.save()
                                sys.exit()

                            elif self.game.sidebar_cursor_current == "play": # sur le boutton jouer
                                self.game.sound_manager.play("mouse_click")
                                if self.state == "settings":
                                    self.state = self.state_save

                                elif not self.game.not_ended or self.state == "choosing_bot":
                                    self.state = "choosing_mod"
                                    self.game.reset_board()

                                else:
                                    self.state = "playing"

                            elif self.game.sidebar_cursor_current == "settings": # sur le boutton paramètres
                                self.game.sound_manager.play("mouse_click")
                                if self.state == "settings":
                                    self.state = self.state_save
                                else:
                                    self.state_save = self.state
                                    self.state = "settings"

                        elif self.state == "showing_results" and self.game.results_review.collidepoint(event.pos): # boutton de revu de la partie
                            self.game.sound_manager.play("mouse_click")
                            self.state = "reviewing"

                        elif self.state == "showing_results" and self.game.results_back_menu.collidepoint(event.pos): # boutton pour revenir au menu de choix du mode
                            self.game.sound_manager.play("mouse_click")
                            self.game.reset_board()
                            self.state = "choosing_mod"

                        elif self.state == "showing_results" and self.game.results_new_game.collidepoint(event.pos): # boutton pour relancer une partie
                            self.game.sound_manager.play("mouse_click")
                            self.state = "playing"
                            self.game.reset_board(newgame=True)

                        elif self.state in ["playing", "reviewing"] and self.game.scroll_bar.collidepoint(event.pos): # barre de défilement
                            self.game.scroll_bar_delta = self.game.scroll_bar_y - self.game.mouse_y
                            self.game.scroll_bar_grabbing = True

                        elif self.state in ["playing", "reviewing"] and self.game.moves_following.collidepoint(event.pos)  and len(self.game.moves) > 1: # boutton de choix d'une position de la partie
                            if self.game.moves_following_index == len(self.game.moves) - 1 and self.game.not_ended:
                                self.state = "playing"
                            else:
                                self.state = "reviewing"
                            self.game.board_shown_number = self.game.moves_following_index
                            self.game.board_shown = self.game.transitions[self.game.board_shown_number][0]
                            self.game.player_turn = self.game.transitions[self.game.board_shown_number][1] if not self.game.not_ended else self.game.player_turn
                            self.game.last_move_rect = self.game.transitions[self.game.board_shown_number][2]
                            self.game.returned_pieces = self.game.transitions[self.game.board_shown_number][3]
                            self.game.is_prepared = False
                            self.game.unplayable_draw = False
                            self.game.unplayable_cnt = 5
                            self.game.sound_manager.play("mouse_click")

                        elif self.state in ["playing", "reviewing"] and self.game.last_one_rect.collidepoint(event.pos)  and len(self.game.moves) > 1: # boutton de retour arrière
                            self.state = "reviewing"
                            self.game.board_shown_number = max(self.game.board_shown_number - 1, 0)
                            self.game.board_shown = self.game.transitions[self.game.board_shown_number][0]
                            self.game.player_turn = self.game.transitions[self.game.board_shown_number][1] if not self.game.not_ended else self.game.player_turn
                            self.game.last_move_rect = self.game.transitions[self.game.board_shown_number][2]
                            self.game.returned_pieces = self.game.transitions[self.game.board_shown_number][3]
                            self.game.is_prepared = False
                            self.game.unplayable_draw = False
                            self.game.unplayable_cnt = 5
                            self.game.sound_manager.play("mouse_click")

                        elif self.state in ["playing", "reviewing"] and self.game.next_one_rect.collidepoint(event.pos)  and len(self.game.moves) > 1:  # boutton de retour avant
                            self.game.board_shown_number = min(self.game.board_shown_number + 1, len(self.game.moves) - 1)
                            if self.game.board_shown_number == len(self.game.moves) - 1 and self.game.not_ended:
                                self.state = "playing"
                            else:
                                self.state = "reviewing"
                            self.game.board_shown = self.game.transitions[self.game.board_shown_number][0]
                            self.game.player_turn = self.game.transitions[self.game.board_shown_number][1] if not self.game.not_ended else self.game.player_turn
                            self.game.last_move_rect = self.game.transitions[self.game.board_shown_number][2]
                            self.game.returned_pieces = self.game.transitions[self.game.board_shown_number][3]
                            self.game.is_prepared = False
                            self.game.unplayable_draw = False
                            self.game.unplayable_cnt = 5
                            self.game.sound_manager.play("mouse_click")

                        elif self.state in ["playing", "reviewing"] and self.game.white_flag_rect.collidepoint(event.pos):
                            self.state = "resigning"
                            self.game.sound_manager.play("mouse_click")

                        elif self.state == "resigning" and (self.game.white_flag_rect.collidepoint(event.pos) or self.game.resign_false.collidepoint(event.pos)):
                            self.state = "playing"
                            self.game.sound_manager.play("mouse_click")

                        elif self.state == "resigning" and self.game.resign_true.collidepoint(event.pos):
                            self.game.do_results(resign=True)
                            self.game.sound_manager.play("mouse_click")

                        elif self.state == "settings" and sqrt((self.game.mouse_x - self.game.settings_fps_point_x)**2 +
                            (self.game.mouse_y - self.game.settings_fps_bar.center[1])**2) <= self.game.settings_fps_point_radius:
                            self.game.settings_fps_point_grabbing = True
                            self.game.settings_fps_points_delta = self.game.settings_fps_point_x - self.game.mouse_x

                        elif self.state == "settings" and sqrt((self.game.mouse_x - self.game.settings_volume_point_x)**2 +
                            (self.game.mouse_y - self.game.settings_volume_bar.center[1])**2) <= self.game.settings_volume_point_radius:
                            self.game.settings_volume_point_grabbing = True
                            self.game.settings_volume_points_delta = self.game.settings_volume_point_x - self.game.mouse_x

                        elif self.state == "settings" and self.game.settings_timer_minus_back.collidepoint(event.pos):
                            self.memory.datas["timer"] = max(0, self.memory.datas["timer"] - 1)
                            self.game.settings_timer_value_text = self.game.settings_values_font.render(f'{self.game.settings_timer_list[self.memory.datas["timer"]]}{":00" if self.memory.datas["timer"] > 0 else ""}', 1, (255, 255, 255))
                            self.game.settings_timer_value_text_rect = self.game.settings_timer_value_text.get_rect(center=(self.game.settings_volume_bar.center[0] + 10, self.game.settings_timer_text_rect.center[1]))
                            self.game.sound_manager.play("mouse_click")

                        elif self.state == "settings" and self.game.settings_timer_plus_back.collidepoint(event.pos):
                            self.memory.datas["timer"] = min(len(self.game.settings_timer_list) - 1, self.memory.datas["timer"] + 1)
                            self.game.settings_timer_value_text = self.game.settings_values_font.render(f'{self.game.settings_timer_list[self.memory.datas["timer"]]}{":00" if self.memory.datas["timer"] > 0 else ""}', 1, (255, 255, 255))
                            self.game.settings_timer_value_text_rect = self.game.settings_timer_value_text.get_rect(center=(self.game.settings_volume_bar.center[0], self.game.settings_timer_text_rect.center[1]))
                            self.game.sound_manager.play("mouse_click")

                        elif self.state == "settings":
                            self.game.sound_manager.play("mouse_click")
                            for name in self.game.settings_board_buttons.keys():
                                if self.game.settings_board_buttons[name][2].collidepoint(event.pos):
                                    self.memory.datas["board"] = name
                                    self.game.game_board = self.game.game_board_dict[self.memory.datas["board"]][0]  # plateau de jeu
                                    self.game.last_move.fill(self.game.game_board_dict[self.memory.datas["board"]][1])
                                    for piece in self.game.returned_pieces.values():
                                        piece[0].fill(self.game.game_board_dict[self.memory.datas["board"]][2])

                        elif (self.state == "playing" and self.game.is_prepared and self.game.waiting_for == 'human'
                              and self.game.game_board_rect.collidepoint((self.game.mouse_x, self.game.mouse_y))):  # entrain de jouer
                            board_x = int((self.game.mouse_y - self.game.game_board_y) // 100) # x pour les colonnes
                            board_y = int((self.game.mouse_x - self.game.game_board_x) // 100) # y pour les lignes
                            in_board = True if 0 <= (board_x or board_y) < 8 else False # simple vérification supplémentaire
                            self.game.human_turn(board_x, board_y) if in_board else None

                        else:
                            self.game.sound_manager.play("mouse_click")


                    elif event.type == pygame.MOUSEBUTTONUP:
                        if self.game.scroll_bar_grabbing: # lache la barre de défilement du suivit de partie
                            self.game.scroll_bar_grabbing = False

                        elif self.game.settings_fps_point_grabbing:
                            self.game.settings_fps_point_grabbing = False

                        elif self.game.settings_volume_point_grabbing:
                            self.game.settings_volume_point_grabbing = False

                    elif event.type == pygame.KEYDOWN: # presse une touche
                        if event.key == pygame.K_F11:
                            if self.fullscreen:
                                self.screen_resized = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
                            else:
                                self.screen_resized = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                            self.fullscreen = not self.fullscreen
                        self.game.pressed[event.key] = True

                    elif event.type == pygame.KEYUP: # arrete de presser une touche
                        self.game.pressed[event.key] = False

                self.screen_resized.blit(pygame.transform.scale(self.screen, (self.screen_resized.get_width(), self.screen_resized.get_height())), (0, 0))
                pygame.display.update()  # mise à jour de l'affichage

                self.clock.tick(self.fps_method(fps=self.memory.datas["fps_max"])) # limite des fps pour éviter les grosses différences d'animation entre les ordinateurs
        finally:
            self.memory.save()

    def init_training(self):
        pygame.quit()
        self.n = int(input("Nombre de parties : "))

        ddqn_score = 0
        games_counter = 0

        for _ in range(self.n):
            self.game.reset_board()

            self.ddqn.episode += 1  # mise à jour du compteur de parties de DQN
            self.ddqn.epsilon = self.ddqn.get_epsilon()
            print(f"DDQN épisode : {self.ddqn.episode}")
            print(f"Epsilon : {self.ddqn.epsilon}")

            # lancement d'une partie
            self.winner, self.winner_score, self.loser_score = self.start_game()
            print(f"Vainqueur : {self.winner} , {self.winner_score} - {self.loser_score}")

            reward = 0.2 if self.winner == self.game.ddqn_number else -0.2 if self.winner == -self.game.ddqn_number else 0

            for i, transition in enumerate(self.ddqn.temp_memory):
                discount = 0.95 ** (len(self.ddqn.temp_memory) - i - 1)  # plus c'est vers la fin, plus c’est proche de 1
                adjusted_reward = round(min(max(transition[2] + reward * discount, -1), 1), 2)

                transition[2] = adjusted_reward
                self.game.rewards.append(adjusted_reward)
                self.ddqn.memory.push(transition)

            self.ddqn.logging_rewards.append(np.mean(np.array(self.game.rewards)))
            self.ddqn.temp_memory = []

            # enregistrement des scores
            if self.winner == self.game.ddqn_number:
                ddqn_score += 1
                self.ddqn.logging_scores.append(self.winner_score)

            elif self.winner == -self.game.ddqn_number:
                self.ddqn.logging_scores.append(self.loser_score)

            else:
                self.ddqn.logging_scores.append(
                    self.winner_score if self.game.ddqn_number == -1 else self.loser_score)

            games_counter += 1

            # sauvegarde externe du modèle DQN toutes les 1000 parties
            if self.ddqn.episode % 1000 == 0 or self.ddqn.episode == 1:
                self.ddqn.save_model()  # sauvegarde du modèle
                self.ddqn.memory.save()  # sauvegarde du buffer

            # actualisation du modèle adverse
            if self.ddqn.episode % 2500 == 0:
                self.old_dqn.main_network.load_state_dict(self.ddqn.target_network.state_dict())
                self.old_dqn.epsilon = self.ddqn.epsilon
                self.old_dqn.memory = self.ddqn.memory

        # affichage du taux de victoire du modèle dqn après la session d'entrainement
        print(f"Taux de victoire: {ddqn_score / games_counter * 100:.2f}%")  # limite du nombre de décimales à 2

        # affichage du/des graphique(s)
        self.plot_stats()

    # fonction gérant le/les graphique(s)
    def plot_stats(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))  # création de la figure

        # Graphique 1 : L'évolution des scores

        # pour une meilleure visualisation, on réduit l'oscillation des scores en utilisant des moyennes de séquences de scores
        scores_array = np.array(self.ddqn.logging_scores)  # transormation liste -> matrice numpy
        scores_bloc_size = min(500, max(1, self.n // 10))  # nombre de données par moyenne
        scores_means = np.array([np.mean(scores_array[i:i + scores_bloc_size]) for i in range(0, len(scores_array), scores_bloc_size)])

        # paramètres du graphique
        plt.subplot(1, 2, 1)
        plt.ylim(0, 64)
        plt.plot(scores_means, color="b", label="Moyenne des scores de l'agent", marker="o")
        plt.title("Scores de l'agent au fil des parties")
        plt.xlabel(f"Moyennes de {scores_bloc_size} parties")
        plt.ylabel("Score")

        # Graphique 2 : L'évolution du taux de victoire
        winrate_bloc_size = min(500, max(1, self.n // 10))  # nombre de données par moyenne
        winrate_array = np.array(self.ddqn.logging_scores)  # transormation liste -> matrice numpy
        winrate_array = np.where(winrate_array > 32, 1, np.where(winrate_array == 32, 0.5, 0))
        winrate_means = np.array([np.mean(winrate_array[i:i + winrate_bloc_size]) for i in range(0, len(winrate_array), winrate_bloc_size)])

        # paramètres du graphique
        plt.subplot(1, 2, 2)
        plt.ylim(0, 1)
        plt.plot(winrate_means, color="r", label="Moyenne des pourcentages de victoire de l'agent", marker="o")
        plt.title("Pourcentages de victoire de l'agent au fil des parties")
        plt.xlabel(f"Moyennes de {winrate_bloc_size} parties")
        plt.ylabel("Taux de victoire")

        # Graphique 3 : L'évolution des récompenses

        # pour une meilleure visualisation, on réduit l'oscillation des scores en utilisant des moyennes de séquences de scores
        rewards_array = np.array(self.ddqn.logging_rewards)
        rewards_bloc_size = min(500, max(1, len(self.ddqn.logging_rewards) // 10))  # nombre de données par moyenne
        rewards_means = np.array(
            [np.mean(rewards_array[i:i + rewards_bloc_size]) for i in range(0, len(rewards_array), rewards_bloc_size)])

        # paramètres du graphique
        plt.figure(figsize=(7, 5))
        plt.subplot(1, 1, 1)
        plt.plot(rewards_means, color="g", label="Moyenne des Récompenses", marker="x")
        plt.title("Récompenses Moyennes au fil des parties")
        plt.xlabel(f"Moyennes de {rewards_bloc_size} parties")
        plt.ylabel("Récompense Moyenne")

        # tracé du graphique
        plt.legend()
        plt.show()

    def start_game(self):
        playing = True
        while playing:
            self.game.mixed_board, self.game.return_board = self.game.check_valid_play(self.game.board, self.game.player_turn)

            if 2 in self.game.mixed_board:  # vérifie qu'on coup au moins est jouable
                # détermination du joueur du tour
                if self.game.bot == "minmax":
                    self.game.ddqn_turn(self.game.mixed_board, self.game.return_board) if self.game.player_turn == self.game.ddqn_number \
                        else self.game.minmax_turn(self.game.return_board)
                else:
                    self.game.ddqn_turn(self.game.mixed_board, self.game.return_board) if self.game.player_turn == self.game.ddqn_number \
                        else self.game.old_dqn_turn(self.game.mixed_board, self.game.return_board)

                self.game.cant_play = False

            elif self.game.cant_play:
                self.game.black_score = np.sum(self.game.board == -1)  # calcul du score noir
                self.game.white_score = np.sum(self.game.board == 1)  # calcul du score blanc

                delta = self.game.black_score - self.game.white_score  # calcul de la différence des deux scores

                if delta != 0:
                    winner = -1 if delta > 0 else 1
                    winner_score = self.game.black_score if winner == -1 else self.game.white_score
                    loser_score = self.game.white_score if winner == -1 else self.game.black_score

                else:
                    winner = 0
                    winner_score = self.game.black_score
                    loser_score = self.game.white_score

                return winner, winner_score, loser_score

            else:
                self.game.cant_play = True
                print(f"{self.game.player_turn} ne peut pas jouer")

othello = Othello()
othello.start_othello()
