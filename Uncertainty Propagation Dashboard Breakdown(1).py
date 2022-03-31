#!/usr/bin/env python
# coding: utf-8

# # <center>**Uncertainty Propagation Analysis Tutorial**
# 
# ### <center>By Joey Ji

# ## Preface
# 
# The following concepts are explained based on a set of predefined Response Surface Equation in the format shown below:
# 
# ## $Y = {\beta}_{0} + \sum \limits _{i=1} ^{k} {\beta}_{i}X_{i} + \sum \limits _{i=1} ^{k} {\beta}_{ii}X_{i}^{2} + \sum \limits _{1\le i\le j} ^{k} {\beta}_{ij}X_{i}X_{j}$
# 
# 
# We have the following prediction variables: $Y_{1}$, $Y_{2}$, $Y_{3}$

# With these prediction variables, we form the following three response surface equation with 5 variables and a set of 21 unique coefficients for each prediction:
# 
# ## $Y_{n = 1, 2, 3} = {\beta}_{0} + \sum \limits _{i=1} ^{5} {\beta}_{i}X_{i} + \sum \limits _{i=1} ^{5} {\beta}_{ii}X_{i}^{2} + \sum \limits _{1\le i\le j} ^{5} {\beta}_{ij}X_{i}X_{j}$
# 
# For the sake of this discussion, all coefficients for each prediction's response surface equation are randomly generated for analysis later.

# The first step is to import all the python packages and generate all relevant coefficient for the response surface equation:

# In[1]:


import numpy as np
from numpy import random
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

beta0 = np.array([random.rand(), random.rand(), random.rand()])
betai = np.array([random.rand(5), random.rand(5), random.rand(5)])
betaij = np.array([random.rand(5, 5), random.rand(5, 5), random.rand(5, 5)])


# ## Profiler
# 
# We generate the following design variable and uncertainty variable ranges, settings, and numbers of samples for the uncertainty variable distribution pickings. The next step is to calculate the predictions based on these parameters:

# In[2]:


dsn_bounds = {
    'x1': [0.21, 0.26],
    'x2': [25., 35.],
    'x3': [325., 400.],
    'x4': [0.8, 1.2],
    'x5': [0.8, 1.2]
}

dsn_settings = {
    'x1': 1.05* sum(dsn_bounds['x1'])/2,
    'x2': sum(dsn_bounds['x2'])/2,
    'x3': sum(dsn_bounds['x3'])/2
}
pred_bounds = {
    'Y1': [],
    'Y2': [],
    'Y3': []
}
n_samples = 100
var1_dist = np.random.uniform(0.8, 1.2, n_samples)
var2_dist = np.random.uniform(0.7, 1.3, n_samples)


# Based on the given distribution for the two uncertainty variables, we will use numpy's random.choice to randomly pick variable settings to mimic monte carlo simulation.

# In[3]:


monteCarlo = np.zeros((n_samples, 2))
monteCarlo[:, 0] = np.random.choice(var1_dist, size = n_samples)
monteCarlo[:, 1] = np.random.choice(var2_dist, size = n_samples)


# The next part is a function that generate input based on each design parameter's setting in the format of a numpy array. Based on the specifc line we wish to plot in the profiler plot, there are four different input: 
# 
# * input for the main trace line in each profiler plot
# * input for two trace lines to highlight the uncertainty bounds
# * input to generate the dash lines on each profiler plot to indicate the current design variable parameter setting
# * input for joint distribution

# In[4]:


def makeInput(xidx, yidx, type, xVal):
    
    input = np.ones((n_samples, 5))
    
    for i in range(0, 5):
        temp = 'x' + str(i + 1)
        # make main trace line input
        if type == 'main' and xVal == None:
            if i == xidx:
                bounds = dsn_bounds[temp]
                input[:, i] = np.linspace(bounds[0], bounds[1], n_samples)
            elif i < 3: 
                input[:, i] = input[:, i] * dsn_settings[temp]
            else:
                input[:, i] = input[:, i] * (sum(dsn_bounds[temp])/2)
                # for the main trace line, the uncertain variable is set to be the average value
                
        # make uncertainty traces input
        elif type == 'uncertain' and not(xVal == None):
            if i == xidx:
                input[:, i] = input[:, i] * xVal
            elif i < 3: 
                input[:, i] = input[:, i] * dsn_settings[temp]
            else:
                input[:, i] = monteCarlo[:, i - 3]
                
        # make input for the dashline tracking
        elif type == 'dash' and not(xVal == None):
            if i == xidx:
                input[:, i] = input[:, i] * xVal
            elif i < 3: 
                input[:, i] = input[:, i] * dsn_settings[temp]
            else:
                input[:, i] = input[:, i] * (sum(dsn_bounds[temp])/2)
                
        # make input for the dashline tracking
        elif type == 'joint_dist' and xVal == None:
            if i < 3: 
                input[:, i] = input[:, i] * dsn_settings[temp]
            else:
                input[:, i] = monteCarlo[:, i - 3]
                
        # make input for constraint diagram
        elif type == 'constraint_diagram' and xVal == None:
            if i == 0:
                input[:, i] = input[:, i] * var1_dist[xidx]
            elif i == 1:
                input[:, i] = input[:, i] * var2_dist[yidx]
            elif i == 2:
                input[:, i] = input[:, i] * dsn_settings['x' + str(3)]
            elif i > 2:
                input[:, i] = monteCarlo[:, i - 3]
            
        else:
            raise NameError('Invalid inputs!!! **Check your xVal input')
            
    return input


# Recall the response surface equation we mentioned above:
# 
# ## $Y_{n = 1, 2, 3} = {\beta}_{0} + \sum \limits _{i=1} ^{5} {\beta}_{i}X_{i} + \sum \limits _{i=1} ^{5} {\beta}_{ii}X_{i}^{2} + \sum \limits _{1\le i\le j} ^{5} {\beta}_{ij}X_{i}X_{j}$
# 
# The makeOutput(input, xidx, yidx) is a function that takes in the input and the specific x and y index which indicate which profiler plot's output it's generating

# In[5]:


def makeOutput(input, xidx, yidx):
    
    part1 = np.ones(n_samples) * beta0[yidx]
    part2 = np.dot(input, np.transpose(betai[yidx]))
    part3 = np.ones(n_samples)
    
    for j in range(0, n_samples):

        temp = input[j, :]
        temp = np.tril(np.dot(temp.reshape((5, 1)), temp.reshape((1, 5)))) * betaij[yidx]
        part3[j] = np.sum(temp)
        
    output = part1 + part2 + part3
    
    return output


# This function takes in the results from output and make a list of traces for plotting. As shown below by each line comment, the list contains traces for the main line, uncertainty variables' spread, and the dash lines

# In[6]:


def makeTraces(xidx, yidx):

    traceOutput = []
    # make main trace
    input = makeInput(xidx, yidx, 'main', None)
    output = makeOutput(input, xidx, yidx)
    traceOutput.append(go.Scatter(x=input[:, xidx], y=output, line = dict(shape = 'linear', color = 'rgb(250, 150, 0)')))
    
    # make uncertainty traces
    temp = 'x' + str(xidx + 1)
    bounds = dsn_bounds[temp]
    xVal = np.linspace(bounds[0], bounds[1], n_samples)
    
    ubx = []
    ubMax = []
    ubMin = []
    for i in range(0, n_samples):
        
        input = makeInput(xidx, yidx, 'uncertain', xVal[i])
        output = makeOutput(input, xidx, yidx)
        # traceOutput.append(go.Scatter(x=np.ones(n_samples) * xVal[i], y=output, opacity = 0.75, line = dict(shape = 'linear', color = 'rgb(50, 180, 0)')))
        ubx.append(xVal[i])
        ubMax.append(max(output))
        ubMin.append(min(output))
    traceOutput.append(go.Scatter(x = ubx, y = ubMin, fill = None, mode='lines', line_color = 'green'))
    traceOutput.append(go.Scatter(x = ubx, y = ubMax, fill = 'tonexty', mode='lines', line_color = 'green'))
    # make dash lines
    yList = [min(output), max(output)]
    
    for i in range(0, 3):
        
        trialOutput = makeOutput(makeInput(i, yidx, 'main', None), i, yidx)
        if min(trialOutput) < yList[0]:
            yList[0] = min(trialOutput)
        if max(trialOutput) > yList[1]:
            yList[1] = max(trialOutput)
    
    yVal = makeOutput(makeInput(xidx, yidx, 'dash', dsn_settings[temp]), xidx, yidx)[0]
    
    # make the tracers
    traceOutput.append(go.Scatter(x=dsn_bounds[temp], y = [yVal, yVal], line = dict(shape = 'linear', color = 'rgb(10, 12, 240)', dash = 'dash')))
    traceOutput.append(go.Scatter(x=[dsn_settings[temp], dsn_settings[temp]], y = yList, line = dict(shape = 'linear', color = 'rgb(10, 12, 240)', dash = 'dash')))

    
    return traceOutput


# This portion of the code actually generate the profiler plots. The double for loops iterate through each individual profiler plot starting from the bottom left one. Each profiler is indicated by the i and j indices. These indices are also passed into the makeTraces function to make all the required traces. 
# 
# The flow of information is demonstrated with the diagram below:
# 
# ![Screen%20Shot%202022-03-02%20at%2011.42.50%20PM.png](attachment:Screen%20Shot%202022-03-02%20at%2011.42.50%20PM.png)

# In[7]:


fig = make_subplots(rows=3, cols=3, shared_xaxes=True, shared_yaxes=True, start_cell="bottom-left",
        horizontal_spacing = 0.01, vertical_spacing = 0.01)    
for i in range(0, 3):
    for j in range(0, 3):    
    
        traceList = makeTraces(j, i)
                         
        for trace in traceList:
            fig.add_trace(trace, row=i+1, col=j+1)

        if i == 0:
            fig.update_xaxes(title_text='x' + str(j+1), row=i+1, col=j+1)
        if j == 0:
            fig.update_yaxes(title_text='Y' + str(i+1), row=i+1, col=j+1)

fig.update_layout(height=600, width=800, title_text="Profiler Plots", showlegend= False)


# In[8]:


fig.show()


# ### Key Takeaways:
# 
# * The original three RSEs have five variables ($x_{1}$, $x_{2}$, $x_{3}$, $x_{4}$, $x_{5}$), in which $x_{4}$ and $x_{5}$ are treated as uncertain variables with uniform distributions
# * i.e. profiler plot $x_{1}$ vs. $Y_{1}$, the intersection of the dash lines indicate the design variable setting for variable $x_{1}$. This profiler's input is generated while holding $x_{2}$ and $x_{3}$ at their respective design variable setting and setting $x_{4}$ and $x_{5}$ to be their average values
# * The green stripe over each plot's main trace indicate the influence from the uncertainties
# 

# ## Joint Distribution
# 
# The joint distribution utilized the input&output generating function we have mentioned above. After generating monte carlo simulation results for $x_{4}$ and $x_{5}$, $x_{1}$, $x_{2}$, and $x_{3}$ are evaluated at their respective design variable setting.
# 
# Three joint distribution plots are generated (Y1 vs. Y3), (Y1 vs. Y2), (Y2 vs, Y3)

# In[9]:


n_samples = 500
var1_dist = np.random.uniform(0.8, 1.2, n_samples)
var2_dist = np.random.uniform(0.7, 1.3, n_samples)
monteCarlo = np.zeros((n_samples, 2))
monteCarlo[:, 0] = np.random.choice(var1_dist, size = n_samples)
monteCarlo[:, 1] = np.random.choice(var2_dist, size = n_samples)


# Similar plotting structure is applied here just like the profiler plots. The color scale indicates the density of point distribution in the output space. The actual points are overlayed on top of the color contour to provide transparency of specific data point value
# 

# In[10]:


outputList = []
for i in range(0, 3):
    outputList.append(makeOutput(makeInput(0, i, 'joint_dist', None), 0, i))
# print(outputList)

fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True, start_cell="bottom-left",
        horizontal_spacing = 0.05, vertical_spacing = 0.05)
traceListColor = 'Reds'
traceList = [
    go.Histogram2dContour(
        x = outputList[0],
        y = outputList[2],
        colorscale = traceListColor
        ),
    
    go.Histogram2dContour(
        x = outputList[1],
        y = outputList[2],
        colorscale = traceListColor
        ),
    
    go.Histogram2dContour(
        x = outputList[0],
        y = outputList[1],
        colorscale = traceListColor
        )
]
scatterListColor = 'darkcyan'
scatterList = [
    go.Scatter(
        x = outputList[0],
        y = outputList[2],
        opacity = 0.8,
        mode = 'markers',
        marker = dict(color=scatterListColor, size = 3)
        ),
    go.Scatter(
        x = outputList[1],
        y = outputList[2],
        opacity = 0.8,
        mode = 'markers',
        marker = dict(color=scatterListColor, size = 3)
        ),
    
    go.Scatter(
        x = outputList[0],
        y = outputList[1],
        opacity = 0.8,
        mode = 'markers',
        marker = dict(color=scatterListColor, size = 3)
        )
]
for count, (trace) in enumerate(traceList):
    if count == 0:
        fig.add_trace(trace, row=1, col=1)
        fig.add_trace(scatterList[count], row=1, col=1)
        fig.update_xaxes(title_text='Y' + str(1), row=1, col=1)
        fig.update_yaxes(title_text='Y' + str(3), row=1, col=1)
        
    elif count == 1:
        fig.add_trace(trace, row=1, col=2)
        fig.add_trace(scatterList[count], row=1, col=2)
        fig.update_xaxes(title_text='Y' + str(2), row=1, col=2)
    else:
        fig.add_trace(trace, row=2, col=1)
        fig.add_trace(scatterList[count], row=2, col=1)
        fig.update_yaxes(title_text='Y' + str(2), row=2, col=1)

fig.update_layout(height=500, width=500, title_text="Joint Distribution", showlegend= False)


# In[11]:


fig.show()


# ## Constraint Diagram

# Before we start implement the constraint diagram, we first need to determine the bounds for each prediction variable as well as the confidence interval. We also need to determine $x_{4}$ and $x_{5}$ by doing a monte carlo simulation. 

# In[12]:


pred_bounds = {
    'Y1': [min(outputList[0]), max(outputList[0])],
    'Y2': [min(outputList[1]), max(outputList[1])],
    'Y3': [min(outputList[2]), max(outputList[2])]
}
con_intvl = 0.8

resolution = 10
n_samples = 100
var1_dist = np.linspace(dsn_bounds['x' + str(1)][0], dsn_bounds['x' + str(1)][1], resolution)
var2_dist = np.linspace(dsn_bounds['x' + str(2)][0], dsn_bounds['x' + str(2)][1], resolution)
monteCarlo = np.zeros((n_samples, 2))
monteCarlo[:, 0] = np.random.choice(var1_dist, size = n_samples)
monteCarlo[:, 1] = np.random.choice(var2_dist, size = n_samples)

print(pred_bounds)


# Once the master bounds for each prediction variable is determined, the next step is to determine the two axes of the constraint diagram. For this example, we will choose to use $x_{1}$ and $x_{2}$. $x_{3}$ is set based on its design variable setting. 
# 
# 
# create a function to find traces for each prediction variable's 

# In[23]:


def makeTraceCD(yidx):
    confidenceInt = 80
    traceOutput = []
    contourListLower = np.ones((resolution, resolution))
    contourListUpper = np.ones((resolution, resolution))
    
    for i, (x) in enumerate(var1_dist):
        for j, (y) in enumerate(var2_dist):
            grid_point_spread = makeOutput(makeInput(i, j, 'constraint_diagram', None), None, yidx)
            # point_bounds = [min(grid_point_spread), max(grid_point_spread)]
            lVal, uVal = np.percentile(grid_point_spread, [50 - confidenceInt/2, 50 + confidenceInt/2])
            contourListLower[i, j] = lVal
            contourListUpper[i, j] = uVal
    
    print(str(min(contourListLower.reshape((100)))) + ' and  ' + str(max(contourListLower.reshape((100)))))
    print(str(min(contourListUpper.reshape((100)))) + '  and  ' + str(max(contourListUpper.reshape((100)))))
    lowerBound = (max(contourListLower.reshape((100))) - min(contourListLower.reshape((100)))) * 0.1 + min(contourListLower.reshape((100)))
    upperBound = (max(contourListLower.reshape((100))) - min(contourListLower.reshape((100)))) * 0.9 + min(contourListLower.reshape((100)))
    

    fig = go.Figure()
    
    xlist = np.ones((resolution, resolution))
    ylist = np.ones((resolution, resolution))
    for i in range(0, 10):
        xlist[i, :] = var1_dist
        ylist[i, :] = np.ones(resolution) * var2_dist[i]
        
    traceList = [   
        go.Contour(
            x=var1_dist, y=var2_dist, z=contourListLower,
            contours=dict(
                type='constraint',
                operation='>',
                value= 19000,
            )
        ),
        
        go.Contour(
            x=var1_dist, y=var2_dist, z=contourListLower,
            contours=dict(
                type='constraint',
                operation='>',
                value= 20000,
            )
        ),
        
        go.Contour(
            x=var1_dist, y=var2_dist, z=contourListUpper,
            contours=dict(
                type='constraint',
                operation='<',
                value= 22000,
            )
        ),
        
        go.Contour(
            x=var1_dist, y=var2_dist, z=contourListUpper,
            contours=dict(
                type='constraint',
                operation='<',
                value= 22500,
            )
        )
    ]
    
    fig.add_trace(
        go.Scatter(
            x = xlist.reshape((100)),
            y = ylist.reshape((100)),
            opacity = 0.8,
            mode = 'markers',
            marker = dict(color='darkcyan', size = 3)
        ))
    fig.add_trace(traceList[0])
    fig.add_trace(traceList[1])
    fig.add_trace(traceList[2])
    fig.add_trace(traceList[3])
    
    fig.show()


# In[24]:


makeTraceCD(0)


# In[ ]:


'''contours=dict(
    type='constraint',
    operation='>',
    value= 98000,
),
line_color='rgb(10, 12, 100)',
fillcolor='rgb(20, 100, 10)',
showlegend=False,
name= 'test two',
opacity=0.3,
hoverinfo='none','''
    
    
    
    contours=dict(
    type='constraint',
    operation='<',
    value= 96800,
),
line_color='rgb(10, 12, 100)',
fillcolor='rgb(20, 100, 10)',
showlegend=False,
name= 'test one',
opacity=0.3,
hoverinfo='none',


# In[15]:


import numpy as np
import random
# print([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# print(np.transpose([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
# print(np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1))
# print(np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))

a = np.array([1, 2, 3, 4, 5])
b = np.transpose(a) * a
print(a)
print(a.reshape(5, 1))
a = np.dot(a.reshape(5, 1), a.reshape(1, 5))
print(a)
print(np.tril(a, 0))
print(np.sum(np.tril(a, 0)))

temp = np.zeros((5, 5))
for x in range(0, 5):
    for y in range(0, 5):
        temp[x, y] = random.random()
print(temp)
print(np.tril(a, 0) * temp)

def generateOutput(xidx, yidx, monteCarlo):
    
    input = np.zeros((100, 5))
    for count, (_, bounds) in enumerate(dsn_bounds.items()):
        input[:, count] = np.linspace(bounds[0], bounds[1], 100)
    part1 = np.ones((100, 3))
    part2 = np.dot(input, np.transpose(betai))
    part3 = np.ones((100, 3))

    for i in range(0, 3):

        part1[:, i] = part1[:, i] * beta0[i]

        for j in range(0, 100):

            temp = input[i, :]
            temp = np.tril(np.dot(temp.reshape((5, 1)), temp.reshape((1, 5)))) * betaij[0]
            part3[j, i] = np.sum(temp)
    output = part1 + part2 + part3
    return output





input = np.zeros((100, 5))
for count, (_, bounds) in enumerate(dsn_bounds.items()):
    input[:, count] = np.linspace(bounds[0], bounds[1], 100)
part1 = np.ones((100, 3))
part2 = np.dot(input, np.transpose(betai))
part3 = np.ones((100, 3))

for i in range(0, 3):

    part1[:, i] = part1[:, i] * beta0[i]

    for j in range(0, 100):

        temp = input[i, :]
        temp = np.tril(np.dot(temp.reshape((5, 1)), temp.reshape((1, 5)))) * betaij[0]
        part3[j, i] = np.sum(temp)
output = part1 + part2 + part3



print(dsn_bounds.items())
print(dsn_bounds['x1'])
print(sum(dsn_bounds['x1']))
fig.add_shape( # add a horizontal "target" line
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot", x0=0, x1=1, xref="paper", y0=100000, y1=100000, yref="y")

fig.add_shape( # add a vertical "target" line
    type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot", 
    x0=sum(dsn_bounds[temp])/2, x1=sum(dsn_bounds[temp])/2, y0=min(output[:, i]), y1=max(output[:, i]))

print(str(sum(dsn_bounds[temp])/2) + '*********' + str(min(output[:, i]))  + '*********' + str(max(output[:, i])))




# OG Profiler Plot
fig = make_subplots(rows=3, cols=3, shared_xaxes=True, shared_yaxes=True, start_cell="bottom-left",
        horizontal_spacing = 0.05, vertical_spacing = 0.05)
for i in range(0, 3):
    for j in range(0, 3):
        output = makeOutput(makeInput(j, i, 'main', None), j, i)
        # traceList.append(go.Scatter(x=input[:, j], y=output[:, i]))
        traceList = []
        traceList.append(go.Scatter(x=makeInput(j, i, 'main', None)[:, j], y=output))
        temp = 'x' + str(j + 1)
        xList = dsn_bounds[temp]
        yList = [sum([min(output), max(output)])/2] * 2
        print(str(min(output)) + '*********' + str(max(output)))
        traceList.append(go.Scatter(x=xList, y = yList, line = dict(shape = 'linear', color = 'rgb(10, 12, 240)', dash = 'dash')))
        traceList.append(go.Scatter(x=[sum(xList)/2, sum(xList)/2], y = [min(output), max(output)], line = dict(shape = 'linear', color = 'rgb(10, 12, 240)', dash = 'dash')))
                         
        for trace in traceList:
            fig.add_trace(trace, row=i+1, col=j+1)

        if i == 0:
            fig.update_xaxes(title_text='x' + str(j+1), row=i+1, col=j+1)
        if j == 0:
            fig.update_yaxes(title_text='Y' + str(i+1), row=i+1, col=j+1)

fig.update_layout(height=600, width=800, title_text="Profiler Plots", showlegend= False)
fig.show()



for j, (y) in enumerate(var2_dist):
    for i, (x) in enumerate(var1_dist):
        for k in range(0, 3):
            # print(makeInput(i, j, 'constraint_diagram', None))
            temp = makeOutput(makeInput(i, j, 'constraint_diagram', None), None, k)
            print(len(temp))
        print('******************* new set')


        
bounds = dsn_bounds['x' + str(1)]
xAxis = np.linspace(bounds[0], bounds[1], n_samples)
for C in range(0, 3):
    for x in xAxis:
        input = np.ones((n_samples, 5))
        input[:, 0] = input[:, 0] * x
        bounds = dsn_bounds['x' + str(2)]
        input[:, 1] = np.linspace(bounds[0], bounds[1], n_samples)
        input[:, 2] = input[:, 2] * dsn_settings['x' + str(3)]
        input[:, 3:5] = monteCarlo
        print('*******************')
        output = makeOutput(input, 0, C) # xidx here doesn't matter, we just arbitrarly chose it as a place holder
        print(output)
        
        
temp = np.random.uniform(-1, 1, n_samples)

print(np.percentile(temp, [10, 90]))

