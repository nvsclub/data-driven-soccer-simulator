# Imports
import tkinter
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import lib.draw as draw
import lib.simulator2 as sim

import numpy as np
from random import random
import time

# Initializing Tkinter
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# Defines
img_dir = 'tmp/screen.png'
rules_text = 'Rules:\nYou are given a random starting position.'
rules_text += '\nLeft click with the mouse on the map to pass the ball'
rules_text += '\nPress space to shoot.'
rules_text += '\nPress f to change action (Pass/Blue, Dribble/Orange).'
x, y = random(), random()

# Tracking variables
global rewards_cumulative
rewards_cumulative = 0
global games_played
games_played = 1
global done
done = False
global currently_passing
currently_passing = True

# Draw initial image
draw.pitch()
plt.scatter(x*100, y*100, color = 'C0')
plt.text(0, 105, rules_text, color = 'black')
reward_text =  'Games Finished: ' + str(games_played)
reward_text += '\nTotal Rewards: ' + str(round(rewards_cumulative, 3))
reward_text += '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

# Update screen flow
def update_screen():
    # Update frame
    img = ImageTk.PhotoImage(Image.open(img_dir))
    panel.configure(image=img)
    panel.image = img

# Summon initial agent
agent = sim.Agent(x, y)

# Retrieve mouse information and display it on screen
def get_clicks(event):
    global rewards_cumulative
    global games_played
    global done
    global currently_passing

    # Define color based on action
    action_color = 'C1'
    if currently_passing:
        action_color = 'C0'

    # Standardize coordinates
    xt = (event.x - 240) / 1072
    yt = (event.y - 90) / 690 * -1 + 1 # Inverting coordinates in the end

    # Check if out of bounds action
    if xt > 1 or xt < 0 or yt > 1 or yt < 0:
        plt.text(50, -0.3, 'Out of Bounds!', color = 'black', horizontalalignment='center', verticalalignment='center')

        plt.savefig(img_dir)
        img = ImageTk.PhotoImage(Image.open(img_dir))
        panel.configure(image=img)
        panel.image = img

        return

    # Reset to plot next picture
    plt.clf()

    obs = [agent.x, agent.y]

    # Do action
    action = 2
    if currently_passing:
        action = 1
    next_obs, reward, is_done, action_used = agent.do_action(action, xt=xt, yt=yt, verbose=True)
    rewards_cumulative += reward
    
    # Ask agent to plot actions
    sim.draw_play(agent, plot_now = True, current_action_color=action_color)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

    # End play clause
    if is_done:
        done = True
        plt.text(50, 101, 'Play over. Press Enter to Restart. Press S to save.', color = 'C1', horizontalalignment='center', verticalalignment='center')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# End play by shooting at goal
def shot(event):
    global rewards_cumulative
    global games_played

    obs = [agent.x, agent.y]

    # Do action
    next_obs, reward, is_done, action_used = agent.do_action(0, verbose=True)
    rewards_cumulative += reward


    # Reset to plot next picture
    plt.clf()
    
    # Ask agent to plot actions
    sim.draw_play(agent, plot_now = True)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')
    plt.text(50, 101, 'Play over. Press Enter to Restart. Press S to save.', color = 'C1', horizontalalignment='center', verticalalignment='center')
    plt.text(103, 50, 'XG: ' + str(round(reward, 3)), color = 'black')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to reset current play
def restart(event):
    global games_played
    global done
    global currently_passing

    # Increment game counter
    games_played += 1
    done = False

    # Define color based on action
    action_color = 'C1'
    if currently_passing:
        action_color = 'C0'

    # Generate new initial coordinates and reset agent
    x, y = random(), random()
    agent.reset(x, y)

    # Clear figure
    plt.clf()

    # Plot initial point
    draw.pitch()
    plt.scatter(x*100, y*100, color = action_color)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to change action type between pass and dribble
def change_action(event):
    global currently_passing
    global done
    currently_passing = not currently_passing

    # Define color based on action
    action_color = 'C1'
    if currently_passing:
        action_color = 'C0'

    # Reset to plot next picture
    plt.clf()

    # Ask agent to plot actions
    sim.draw_play(agent, plot_now = True, current_action_color=action_color)

    # Additional information and stats to the player
    plt.text(0, 105, rules_text, color = 'black')
    reward_text =  'Games Played: ' + str(games_played) + '\nTotal Rewards: ' + str(round(rewards_cumulative, 3)) + '\nRewards pg: ' + str(round(rewards_cumulative/games_played, 3))
    plt.text(100, 105, reward_text, color = 'black', horizontalalignment='right')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to save current frame displayed
def save(event):
    plt.savefig('img/saves/' + time.asctime().replace(':','') + '.png')

    # Indicate to the player that the play has been saved
    plt.text(50, -10, 'Saved', color='black', horizontalalignment='center', verticalalignment='center')

    # Save frame for update
    plt.savefig(img_dir)

    # Update frame
    update_screen()

# Function to correcly quit interface
def _quit():
    root.quit()
    root.destroy()


# Bind keys to functions
root.bind('<space>', shot)
root.bind('<Return>', restart)
root.bind('f', change_action)
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