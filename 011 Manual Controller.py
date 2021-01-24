# Imports
import tkinter
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import lib.draw as draw
import lib.simulator as sim

import numpy as np
from random import random
import time

# Initializing Tkinter
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# Defines
img_dir = 'tmp/screen.png'
rules_text = 'Rules:\nYou are given a random starting position.\nLeft click with the mouse on the map to pass the ball\nPress space to shoot.'
x, y = random(), random()

# Tracking variables
global rewards_cumulative
rewards_cumulative = 0
global games_played
games_played = 1


# Draw initial image
draw.pitch()
plt.scatter(x*100, y*100, color = 'C0')
plt.text(0, 105, rules_text, color = 'white')
reward_text =  'Games Finished: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
plt.text(100, 105, reward_text, color = 'white', horizontalalignment='right')

# Summon initial agent
agent = sim.Agent(x, y)

# Retrieve mouse information and display it on screen
def get_clicks(event):
    global rewards_cumulative
    global games_played

    print(event.x, event.y)
    
    # Standardize coordinates
    xt = (event.x - 240) / 1072
    yt = (event.y - 90) / 690 * -1 + 1 # Inverting coordinates in the end

    # Check if out of bounds action
    if xt > 1 or xt < 0 or yt > 1 or yt < 0:
        plt.text(50, -0.3, 'Out of Bounds!', color = 'white', horizontalalignment='center', verticalalignment='center')

        plt.savefig(img_dir)
        img = ImageTk.PhotoImage(Image.open(img_dir))
        panel.configure(image=img)
        panel.image = img

        return

    # Reset to plot next picture
    plt.clf()

    obs = [agent.x, agent.y]

    # Do action
    next_obs, reward, is_done, action_used = agent.do_action(1, xt=xt, yt=yt, verbose=True)
    rewards_cumulative += reward

    # Register for debug
    f = open('data_tracker.csv', 'a')
    f.write(str(obs) + ',' + str(action_used) + ',' + str(reward) + '\n')
    f.close()
    
    # Ask agent to plot actions
    sim.draw_play(agent, plot_now = True)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'white')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'white', horizontalalignment='right')

    # End play clause
    if is_done:
        plt.text(50, 101, 'Play over. Press Enter to Restart. Press S to save.', color = 'yellow', horizontalalignment='center', verticalalignment='center')

    # Update frame
    plt.savefig(img_dir)
    img = ImageTk.PhotoImage(Image.open(img_dir))
    panel.configure(image=img)
    panel.image = img

# End play by shooting at goal
def shot(event):
    global rewards_cumulative
    global games_played

    obs = [agent.x, agent.y]

    # Do action
    next_obs, reward, is_done, action_used = agent.do_action(0, verbose=True)
    rewards_cumulative += reward

    # Register for debug
    f = open('data_tracker.csv', 'a')
    f.write(str(obs) + ',' + str(action_used) + ',' + str(reward) + '\n')
    f.close()

    # Reset to plot next picture
    plt.clf()
    
    # Ask agent to plot actions
    sim.draw_play(agent, plot_now = True)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'white')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'white', horizontalalignment='right')
    plt.text(50, 101, 'Play over. Press Enter to Restart. Press S to save.', color = 'yellow', horizontalalignment='center', verticalalignment='center')
    plt.text(103, 50, 'XG: ' + str(round(reward, 3)), color = 'white')

    # Update frame
    plt.savefig(img_dir)
    img = ImageTk.PhotoImage(Image.open(img_dir))
    panel.configure(image=img)
    panel.image = img

# Function to reset current play
def restart(event):
    # Increment game counter
    global games_played
    games_played += 1

    # Generate new initial coordinates and reset agent
    x, y = random(), random()
    agent.reset(x, y)

    # Clear figure
    plt.clf()

    # Plot initial point
    draw.pitch()
    plt.scatter(x*100, y*100, color = 'C0')

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'white')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'white', horizontalalignment='right')

    # Update frame
    plt.savefig(img_dir)
    img = ImageTk.PhotoImage(Image.open(img_dir))
    panel.configure(image=img)
    panel.image = img

# Function to save current frame displayed
def save(event):
    plt.savefig('img/saves/' + time.asctime().replace(':','') + '.png')

    # Indicate to the player that the play has been saved
    plt.text(50, -10, 'Saved', color='white', horizontalalignment='center', verticalalignment='center')

    # Update frame
    plt.savefig(img_dir)
    img = ImageTk.PhotoImage(Image.open(img_dir))
    panel.configure(image=img)
    panel.image = img

# Function to correcly quit interface
def _quit():
    root.quit()
    root.destroy()


# Bind keys to functions
root.bind('<space>', shot)
root.bind('<Return>', restart)
root.bind('s', save)

# Get full screen
root.attributes("-fullscreen", True)

# Prints first image
plt.savefig(img_dir)
img = ImageTk.PhotoImage(Image.open(img_dir))
panel = tkinter.Label(root, image=img)
panel.bind('<Button-1>', get_clicks)
panel.pack()

# Add quit button
button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

# Enter mainloop
tkinter.mainloop()