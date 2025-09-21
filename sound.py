import pygame


class SoundManager:

    def __init__(self, game):
        self.game = game
        self.special_channel = pygame.mixer.Channel(0)
        pygame.mixer.set_reserved(1)  # réserve le premier canal

        self.sounds = {
            "place" : (pygame.mixer.Sound(self.game.main.get_path("assets/sounds/place.ogg")), 1.00),
            "winning": (pygame.mixer.Sound(self.game.main.get_path("assets/sounds/winning.ogg")), 1.00),
            "game_start": (pygame.mixer.Sound(self.game.main.get_path("assets/sounds/game_start.ogg")), 0.75),
            "mod_choosen": (pygame.mixer.Sound(self.game.main.get_path("assets/sounds/mod_choosen.ogg")), 0.10),
            "mouse_click": (pygame.mixer.Sound(self.game.main.get_path("assets/sounds/mouse_click.ogg")), 0.10),
            "unplayable_move": (pygame.mixer.Sound(self.game.main.get_path("assets/sounds/unable_move.ogg")), 0.10),
        }

    def play(self, name):
        self.sounds[name][0].set_volume(self.game.main.memory.datas["volume"] * self.sounds[name][1] / 100)

        if name in ["winning", "mod_choosen"]:
            self.special_channel.play(self.sounds[name][0])
        else:
            self.sounds[name][0].play()

    def music(self, name):
        pygame.mixer.music.load(self.game.main.get_path(f"assets/sounds/{name}.ogg"))
        pygame.mixer.music.play(-1, fade_ms=3000)
        pygame.mixer.music.set_volume(self.game.main.memory.datas["volume"] / 200)
