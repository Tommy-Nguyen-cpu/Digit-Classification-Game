import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import pygame
import numpy as np
import sys
import MLText
import Inference

class DigitDrawingApp:
    def __init__(self, screen_width, screen_height):
        pygame.init()

        self.screen_width = screen_width
        self.screen_height = screen_height

        # Set up Pygame screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Digit Drawing App")

        # Initialize drawing variables
        self.drawing = False
        self.digit_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        self.digit_surface = pygame.surfarray.make_surface(self.digit_image)

        # Run the drawing loop
        self.run()

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.drawing = True
                elif event.type == pygame.MOUSEMOTION and self.drawing:
                    pos = pygame.mouse.get_pos()
                    pygame.draw.circle(self.screen, (255, 255, 255), pos, 10)
                    pygame.draw.circle(self.digit_surface, (255, 255, 255), pos, 10)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False

            # Blit the digit surface onto the screen
            self.screen.blit(self.digit_surface, (0, 0))

            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

    def save_drawing(self, filename):
        # Convert the Pygame surface to a NumPy array
        digit_array = pygame.surfarray.array3d(self.digit_surface)

        # Transposes image.
        digit_array = np.transpose(digit_array, axes=(1,0, 2))

        np.save(filename, digit_array)

    def load_and_display_drawing(self, filename):
        # Load the NumPy array from file
        loaded_array = np.load(filename)
        predict = Inference.Predict(loaded_array)
        print(predict)

        # Display the loaded array using Seaborn
        plt.figure(figsize=(6, 6))

        plt.annotate("Prediction: " + str(predict), [0,0])
        # Display the loaded array using Matplotlib's imshow
        plt.imshow(loaded_array, cmap='gray')  # Transpose to swap axes
        #ME: w/o transpose, image is sideways. 
        plt.axis("off")
        plt.show()

if __name__ == "__main__":

    # MLText.TrainSVC()
    app = DigitDrawingApp(400, 400)
    app.save_drawing("drawn_digit.npy")

    # Load and display the saved drawing using Seaborn
    app.load_and_display_drawing("drawn_digit.npy")

    #Env info:
        # pip install -r requirements.txt
    #or
        # python3 -m venv .digitsvenv
        # source /Users/memo/Documents/digits/.digitsvenv/bin/activate  
        # pip install scikit-learn pygame pandas matplotlib