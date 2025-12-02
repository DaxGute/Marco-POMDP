import os
import time

from MarcoPolo import MarcoPolo
from MarcoPoloPOMDP import MarcoPoloPOMDP

POOLS_DIR = "pools"


def main():
    pool_files = [f for f in os.listdir(POOLS_DIR)]

    print("Available pool shapes:\n")
    for i, filename in enumerate(pool_files):
        print("[" + str(i) + "]" + str(filename))

    choice = input("\nEnter the number of the pool you want to load: ").strip()

    game = MarcoPoloPOMDP(pool_files[int(choice)], num_polos=3, diagnostics=[True, True, False, False])

    game.render()

    while not game.iterate_round():
        game.render()
        
        game.display_diagnostics()

        i += 1

        input("Press Enter to continue...")
    game.render()

    print("Game over!")
    print("Marco was caught by a polo!")


if __name__ == "__main__":
    main()

