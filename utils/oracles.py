import math

def reward_area(obs, area):
    goal_x = (area[0] + area[2]) / 2
    goal_y = (area[1] + area[3]) / 2 
    if obs[0] > area[0] and obs[0] < area[2] and obs[1]> area[1]  and obs[1] < area[3]:
        return 1
    else:
        return - math.sqrt(pow((goal_x - obs[0]), 2) + pow(goal_y - obs[1], 2))