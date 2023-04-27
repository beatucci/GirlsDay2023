##########################################
# Interactive website for Girls Day 2023 #
##########################################

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator

# set fav icon & page title
st.set_page_config(page_title="Girls Day 2023", page_icon=":milky_way:", layout="centered")

# Read in interpolation and keep in memory
@st.cache_data
def Prepare_Data(N = 192):
    '''
    N: grid size | int
    '''
    
    # load grid interpolation stored at Bea's website
    import pickle 
    import requests
    
    file_loc = 'https://wwwmpa.mpa-garching.mpg.de/~tucci/grid_interpolation.pck'    
    
    r = requests.get(file_loc, stream='True')    
    grid_interp = pickle.load(r.raw)

    # create xyz list to plot the grid at
    xyz_bin = np.linspace(0, N, N + 1, endpoint=True)
    xyz_bin = 0.5 * (xyz_bin[1:] + xyz_bin[:-1])    
    
    xyz_grid = np.meshgrid(xyz_bin, xyz_bin, xyz_bin, indexing='ij')
    xyz_list = np.reshape(xyz_grid, (3, -1), order='C').T    
        
    return grid_interp, xyz_list

# utility function to plot text
def Annotation(title, x, y = 1.05, size = 19):
    '''
    title: text to be plotted | string
    x: position on x-axis | float
    '''
    
    annotations = dict(xref='paper', yref='paper', x=x, y=y, xanchor='center', yanchor='bottom', text=title, font=dict(family=None, size=size, color='white'), showarrow=False) 
    
    return annotations

def Get_Labels(N = 192, L = 2000.):
    '''
    N: grid size | int
    L: box size [Mpc/h] | int
    '''
        
    xyz_bin = np.linspace(0, L, N + 1, endpoint=True)
    
    # Mpc/h -> billion light years
    xyz = 3.262e-3 * 0.5 * (xyz_bin[1:] + xyz_bin[:-1]) / 0.7
    
    return xyz.astype('str') #labels

def Plot_Galaxies():
    
    hubble = 'data/hubble.jpg'
    density = 'data/density_assignment.png'
    
    col1, col2, col3 = st.columns([1,0.2,1])
    
    col1.image(hubble, caption='Credits: ESO')
    
    col2.header("")
    col2.header("")
    col2.header("")
    col2.header(" →")
    
    col3.image(density)

# Define a function to generate a new grid given alpha 
def Grid(alpha, N = 192, index = 0): 
    '''
    alpha: alpha value for which the interpolation will be evaluated | float
    N: grid size | int
    '''
    
    return grid_interp((np.hstack((alpha * np.ones(len(xyz_list))[:,np.newaxis], xyz_list)))).reshape(N,N,N)[:,:,index]
    
# Plot ground truth grid    
def Plot_Grid_True(grid_true, colormap = 'inferno', zmin = -2, zmax = 10):
    '''
    grid_true: grid evaluated at alpha_true | float
    index: z-axis slice to be plotted | int
    colormap: color map for colorbar | string
    zmin: minimum alpha value for colorbar | float
    zmax: maximum alpha value for colorbar | float    
    '''   
    
    labels = Get_Labels()
    # plot the slice
    fig = px.imshow(grid_true + 1, zmin = zmin, zmax = zmax, color_continuous_scale = colormap, binary_string=False, x = labels, y = labels, 
                    labels = dict(x="billion light-years", y="billion light-years", color=None))
    
    fig.update_layout(annotations=[Annotation("Observation", 0.5)])
    
    st.write(fig)
        
# Plot ground truth grid    
def Plot_Grid_Interactive(grid_true, grid_interactive, colormap = 'inferno'):
    '''
    grid_true: grid evaluated at alpha_true | float
    index: z-axis slice to be plotted | int   
    colormap: color map for colorbar | string
    '''  

    labels = Get_Labels()

    # add observation side by side with simulation
    fig = make_subplots(1, 2)

    fig1 = px.imshow(grid_true + 1, x = labels, y = labels)
    fig2 = px.imshow(grid_interactive + 1, x = labels, y = labels)

    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)

    fig.update_layout(annotations = [Annotation("Observation", 0.225), Annotation("Simulation", 0.775), Annotation("billion light-years", 0.775, -0.225, 14)], coloraxis = {'colorscale' : colormap}, xaxis_title="billion light-years", yaxis_title="billion light-years")

    st.write(fig)
    
# define 'field-level' metric bewteen ground truth and alpha choice
def Plot_Field_Metric(grid_true, grid_interactive):
    '''
    grid_true: grid evaluated at alpha_true | float
    grid_interactive: grid evaluated at selected alpha | float
    '''  
    
    # calculate the metric
    metric = np.sum((grid_true - grid_interactive)**2)
    
    # save the metric
    st.session_state.metric.append(metric)
    
    # plot metric vs alpha
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(st.session_state.alpha)*sig_true, y=st.session_state.metric, mode='markers', name='markers', marker=dict(color="firebrick", size=8)))
    
    fig.update_layout(xaxis_title='sigma_8', yaxis_title='Metric', yaxis = dict(tickfont = dict(size=15), title_font = {"size": 25}), xaxis = dict(tickfont = dict(size=20), title_font = {"size": 25}))
    
    st.write(fig)
    
# load interpolation
grid_interp, xyz_list = Prepare_Data()

# set session states to keep variables saved
variables_saved = ['alpha', 'metric']

for variable in variables_saved:
    if variable not in st.session_state:
        st.session_state[variable] = []       
    
#################
# Set variables #
#################

# colormap used for plotting
colormap = 'inferno'
# slice to be plotted
index = 0
# ground truth alpha
alpha_true = 1.

#####################
# Page construction #
#####################

# title
st.title("The Universe at Our Fingertips :milky_way:")
st.markdown("_The distribution of galaxies in the sky contains precious information about the underlying laws of Physics. How can we extract this information to better understand our Universe?_")

st.divider()

st.markdown(":telescope: Suppose that you are observing a region of the night sky and that you decide to count how many galaxies there are out there. Not only that, you want to check how the positions of these galaxies are distributed across the sky. To make your life easier, you draw a grid and count how many galaxies lie inside each of its squares.")

Plot_Galaxies()

st.markdown("You can notice that **their distribution is almost random, but not completely**: there are some pieces of the sky where you can find more galaxies than others. The brightest spots correspond to regions where you detected more galaxies, while the darkest ones are associated to the emptier regions of our Universe.")

# get the grid
if ("grid_true" not in st.session_state):
    grid_true = Grid(alpha_true, index = index)
    st.session_state.grid_true = grid_true

Plot_Grid_True(grid_true = st.session_state.grid_true, colormap = colormap, zmin = None, zmax = None)

st.markdown(":female-technologist: As a cosmologist, you want to find the fundamental laws of Physics based on the observations you made. We believe that the universe started out very smooth, with only tiny fluctuations in the density. But as time goes on, these tiny fluctuations grow by their self-gravity. One thing you want to discover is the value of the parameter $\sigma_8$, which tells you how small the initial fluctuations where typically.")
st.markdown("You therefore create a simulation in your computer that predicts how galaxies would be distributed on the same grid for different values of $\sigma_8$. Then you can find what the value of $\sigma_8$ in the Universe is by searching for the simulation that is most similar to your observation.")

# alpha slider bar for interactive grid
sig_true = 0.8
alpha = st.slider("sigma_8", min_value = 0.5*sig_true, max_value = 2.*sig_true, step = 0.1, value = 2.*sig_true)

# save selected alpha
st.session_state.alpha.append(alpha/sig_true)

# evaluate the grid at selected alpha
grid_interactive = Grid(alpha, index = index)

Plot_Grid_Interactive(grid_true = st.session_state.grid_true, grid_interactive = grid_interactive, colormap = colormap)

# graphic with metric between true and interactive choice
Plot_Field_Metric(grid_true = st.session_state.grid_true, grid_interactive = grid_interactive)

st.header("")
st.header("")

st.markdown(":female-teacher: Actually, you want to be a bit more quantitative when you say how similar a simulation is to your observation, so that you can defend the value you have chosen for $\sigma_8$ in front of your peers. You therefore define a _distance metric_, a single, positive number, that gets smaller and smaller the more alike your simulations and observations are.")

st.text("")
# st.latex(r'''\text{Metric} = \sum_{\text{grid}} \Big(\text{Simulation}(\alpha) - \text{Observation}\Big)^2''')
st.latex(r'''\text{Metric} = \Big(\text{Simulation} - \text{Observation}\Big)^2''')
st.text("")

st.subheader("Can you find the value of $\sigma_8$ that minimizes this **metric**? ")

# give the best value for alpha
alpha_chosen = st.text_input("Answer:")

# success or error message
if (alpha_chosen != ""):
    try:
        if (float(alpha_chosen)/sig_true == float(alpha_true)):
            st.success("Congratulations! You found the correct value of $\sigma_8$!", icon="✅")
            st.caption("You have experienced a bit of our day-to-day life studying the Large-Scale-Structure of the Universe (i.e., the study of how the structure of our Universe looks like on very large scales). From the observations of galaxies, we can also find several other _cosmological parameters_ besides $\sigma_8$. All these parameters can help us with some unsolved puzzles in Physics, such as how did the Universe begin, to where it is going to evolve, what is dark matter and dark energy, and even give us hints about the particle physics world, such as neutrinos.")
        
        else:
            st.error("Almost there... try some more values of $\sigma_8$! (Tip: you can select a region of the graphic to zoom in)")     
               
    except:
        st.error("Please enter a number.")
    
st.header("")
st.divider()

# footer
col1, col2, col3 = st.columns([2,1,1])

col1.text("")
col1.markdown("© _Beatriz Tucci, Julia Stadler, Fabian Schmidt._")
col1.markdown("_Max Planck Institute for Astrophysics._")
col1.markdown("_Girls Day 2023._")

col2.text("")
col2.image('data/MPA_logo.png')

col3.text("")
col3.image('data/LEFTfield_logo.png')

st.text("")
st.caption("[Imprint](https://www.mpa-garching.mpg.de/imprint) | [Privacy Policy](https://www.mpa-garching.mpg.de/privacypolicy)")