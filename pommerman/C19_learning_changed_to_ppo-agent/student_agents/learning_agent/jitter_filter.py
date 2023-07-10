class JitterFilter:
    def __init__(self):
        self.y_positions = []
        self.x_positions = []

    def reset(self):
        self.y_positions = []
        self.x_positions = []

    def has_jitter(self, y_pos, x_pos):
        self.y_positions.append(y_pos)
        self.x_positions.append(x_pos)
        if len(self.x_positions) >= 15:
            static_y = len(set(self.y_positions[-15:])) == 1
            static_x = len(set(self.x_positions[-15:])) == 1
            bicond_odd_y = len(set(self.y_positions[-15::2])) == 1
            bicond_even_y = len(set(self.y_positions[-14::2])) == 1
            bicond_odd_x = len(set(self.x_positions[-15::2])) == 1
            bicond_even_x = len(set(self.x_positions[-14::2])) == 1
            if static_y and static_x or \
                    bicond_odd_y and bicond_even_y and static_x or \
                    bicond_odd_x and bicond_even_x and static_y:
                # take random move - exclude staying
                return True

        return False