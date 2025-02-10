import cupy as cp
import numpy as np
import os
import tempfile
from IPython.display import HTML, display
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tdgl
from tdgl.geometry import box, circle
from tdgl.visualization.animate import create_animation
import tdgl.sources



def check_system():
	# Check if the system is running on a GPU
	if cp.cuda.is_available():
		print("GPU is available")
		print("GPU: ", cp.cuda.runtime.getDeviceProperties(0)['name'])
	else:
		print('GPU is not available')


def make_video_from_solution(
	solution,
	quantities=("order_parameter", "phase"),
	fps=30,
	figsize=(5, 4),
	cmap=None,
	vmin_a=None,
	vmax_a=None,
	cmap_b=None,
	):
	"""Generates an HTML5 video from a tdgl.Solution."""
	with tdgl.non_gui_backend():
		with h5py.File(solution.path, "r") as h5file:
			print(cmap_b)
			anim = create_animation(
				h5file,
				quantities=quantities,
				fps=fps,
				figure_kwargs=dict(figsize=figsize, dpi=100),
				vmin=vmin_a,
				vmax=vmax_a,
				cmap=cmap_b,
			)
			video = anim.to_html5_video()
		return HTML(video)
	


def define_4_terminal_mesh(width=0.8, height=0.8, length_units="um", xi=0.1, london_lambda=2, d=0.1, probes=None,terminal_width=0.05, terminal_height=0.1):


	# Material parameters

	xi = 0.1
	london_lambda = 2
	d = 0.1
	layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d, gamma=1)



	film = tdgl.Polygon('film', points=box(width, height)).resample(401).buffer(0)
	
	source1 = tdgl.Polygon('source1', points=box(terminal_width,terminal_height)).translate(-width/2, 0 ).resample(401).buffer(0)
	drain1 = tdgl.Polygon('drain1', points=box(terminal_width,terminal_height)).translate(width/2, 0 ).resample(401).buffer(0)
	source2 = tdgl.Polygon('source2', points=box(terminal_height,terminal_width)).translate(0,-height/2 ).resample(401).buffer(0)
	drain2 = tdgl.Polygon('drain2', points=box(terminal_height,terminal_width)).translate(0,height/2).resample(401).buffer(0)

	if probes==None:
		probes=[((-width*(0.4),-height*(0.4)),
				(width*(0.4),height*(0.4)),
				)]

	device=tdgl.Device("2currents",
				   layer=layer 
				   ,film=film
				   ,terminals= [source1,drain1,source2,drain2]
				   ,length_units=length_units
				   ,probe_points=probes
				   )
	
	return device
def define_2_terminal_mesh(width=0.8, height=0.8, length_units="um", xi=0.1, london_lambda=2, d=0.1, probes=None):


	# Material parameters

	xi = 0.1
	london_lambda = 2
	d = 0.1
	layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d, gamma=1)



	film = tdgl.Polygon('film', points=box(width, height)).resample(401).buffer(0)
	
	source1 = tdgl.Polygon('source1', points=box(0.05,0.1)).translate(-width/2, 0 ).resample(401).buffer(0)
	drain1 = tdgl.Polygon('drain1', points=box(0.05,0.1)).translate(width/2, 0 ).resample(401).buffer(0)

	if probes==None:
		probes=[((-width*(0.4),-height*(0.4)),
					   				(width*(0.4),height*(0.4)),
									)]

	device=tdgl.Device("2currents",
				   layer=layer 
				   ,film=film
				   ,terminals= [source1,drain1]
				   ,length_units=length_units
				   ,probe_points=probes
				   )
	
	return device


def build_mesh(device, max_edge_L=0.025, print=True, plot=True):
	# max_edge_length= xi / 4
	device.make_mesh(max_edge_length=max_edge_L, smooth=100)
	if print:
		device.mesh_stats(print_b=True)
	if plot:
		fig, ax = device.plot(mesh=True, legend=False)
		return fig, ax


def current_series(device, min_current=0,max_current=1,n_steps=100,H_field=10,H_units="mT", current_units="uA",solve_time=200):
	'''Returns a list of solutions for a range of currents
	
	Parameters
	----------
	device : tdgl.Device
		The device to simulate
	min_current : float
		The minimum current to simulate
	max_current : float
		The maximum current to simulate
	n_steps : int
		The number of curretns to simulate
	H_field : float
		The magnetic field to apply 
	H_units : str
		The units of the magnetic field, pyTDGL accepted units
	solve_time : float
		The time to simulate for in taus

	Returns
	-------
	list
		A list of tdgl.Solution objects for each current in the range

	'''
	options = tdgl.SolverOptions(
			solve_time=solve_time,
			# output_file=os.path.join(tempdir.name, "weak-link-zero-current.h5"),
			field_units = H_units,
			current_units=current_units,
			# gpu=True,

		)	

	applied_vector_potential=tdgl.sources.constant.ConstantField(H_field,field_units=options.field_units,length_units=device.length_units)
	currents = np.linspace(min_current,max_current,n_steps)
	solutions = []

	ii=0
	for current_value in currents:
		solution=tdgl.solve(
			device=device,
			options=options,
			applied_vector_potential=applied_vector_potential,
			terminal_currents=dict( source1=current_value, 
						  			drain1=-current_value,
									source2=current_value,
									drain2=-current_value)
		)
		solutions.append(solution)
		ii+=1
		print(f"{ii}/{n_steps} done")
		
	return solutions, currents


