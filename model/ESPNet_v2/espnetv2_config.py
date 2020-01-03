#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

sc_ch_dict = {
            0.5: [16, 32, 64, 128, 256, 1024],
            1.0: [32, 64, 128, 256, 512, 1024],
            1.25: [32, 80, 160, 320, 640, 1024],
            1.5: [32, 96, 192, 384, 768, 1024],
            2.0: [32, 128, 256, 512, 1024, 1280]
        }

rep_layers = [0, 3, 7, 3]

# limits for the receptive field at each spatial level
recept_limit = [13, 11, 9, 7, 5]
branches = 4

# input reinforcement related parameters
config_inp_reinf = 3
input_reinforcement = True