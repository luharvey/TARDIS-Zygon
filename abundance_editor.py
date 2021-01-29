"""
▄▀█ █▄▄ █░█ █▄░█ █▀▄ ▄▀█ █▄░█ █▀▀ █▀▀   █▀▀ █▀▄ █ ▀█▀ █▀█ █▀█
█▀█ █▄█ █▄█ █░▀█ █▄▀ █▀█ █░▀█ █▄▄ ██▄   ██▄ █▄▀ █ ░█░ █▄█ █▀▄

Luke Harvey - 29/01/2021
Trinity College Dublin

Importing this script into a jupyter notebook with ' %matplotlib notebook ' set will allow these functions to run.
An accompanying notebook can be found on my GitHub.
"""

from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt

#Used as the resolution to create the list from which line colours will be extracted
colour_step = 5
#Matplotlib markers and linestyles to cycle through for each of the different elements
markers = ['o']
linestyles = ['-']

#Forming the colours array
def dec_hex(input):
	output = hex(input)[2:]
	if len(output) == 2:
		return output
	else:
		return '0'+output
class rainbow:
	rainbow = []
	for i in (('#ff' + dec_hex(g) + '00') for g in np.arange(0, 251, colour_step)):
		rainbow.append(i)
	for i in (('#' + dec_hex(r) + 'ff' + '00') for r in np.arange(250, -1, -colour_step)):
		rainbow.append(i)
	for i in (('#00ff' + dec_hex(b)) for b in np.arange(5, 251, colour_step)):
		rainbow.append(i)
	for i in (('#00' + dec_hex(g) + 'ff') for g in np.arange(250, -1, -colour_step)):
		rainbow.append(i)
	for i in (('#' + dec_hex(r) + '00ff') for r in np.arange(5, 251, colour_step)):
		rainbow.append(i)
	for i in (('#ff00' + dec_hex(b)) for b in np.arange(250, -1, -colour_step)):
		rainbow.append(i)

#Removes the newline character from the leading line of the file
def clean_newline(array):
	for i in range(len(array)):
		array[i] = array[i].strip()

	if len(array[len(array)-1]) == 0:
		del array[len(array)-1]

	return array

#Switches elements of input array to all lowercase
def lower(array):
	for i in range(len(array)):
		array[i] = array[i].lower()

	return array

#Capitalises the leading letter of each entry in the array, used from the leading line of the export file
def capitalise_array(array):
	for i in range(len(array)):
		array[i] = array[i][0].upper() + array[i][1:]
	return array

#Class to hold the profile array - mass fractions per shell
class element_profile():
	def __init__(self, symbol, shells):
		self.profile = np.zeros(shells)	

	#Function to rewrite the profile array
	def change_profile(self, new):
		self.profile = new

#Overall class to contain the 
class model():
	def __init__(self, symbol_array, shells):
		#Establishing the number of shells as well as an array 'x' for plotting
		self.shells = shells
		self.x = np.arange(1, shells+1, 1)

		#All the 'element_profile' instances will be stored in the dictionary 'data' each with their corresponding symbol
		self.data = {}
		
		#Storing the list of requested elements as a parameter
		self.symbol_array = []
		for i in symbol_array:
			self.symbol_array.append(i.lower())

		#Sampling the 'rainbow.rainbow' array to get an even spread of colours for plotting
		self.colours = []
		self.step = 0.02
		for i in np.arange(0, len(rainbow.rainbow)-1, int(np.ceil(len(rainbow.rainbow)/len(symbol_array)))):
			self.colours.append(rainbow.rainbow[i])

		#Calling the 'add_element' function to populate the dictionary 'data'
		for i in self.symbol_array:
			self.add_element(i)

	#Add element_profile classes to the dictionary 'data'
	def add_element(self, symbol):
		self.data[symbol] = element_profile(symbol, self.shells)

	#Function to display the elemental abundance profiles as a plot
	def plot(self):
		self.fig = plt.figure(num = 'Abundance Profile')
		self.ax = self.fig.add_subplot(1, 1, 1)

		#Each line will be stored in the dictionary 'lines' - similar to the 'data' dictionary they can be called by their corresponding symbol
		self.lines = {}
		for i in range(len(self.symbol_array)):
			self.lines[self.symbol_array[i]], = self.ax.plot(self.x, self.data[self.symbol_array[i]].profile,\
				marker = markers[i%len(markers)], label = self.symbol_array[i], color = self.colours[i],\
				linestyle = linestyles[i%len(linestyles)])

		#Plot housekeeping
		self.ax.set_xlim(1, self.shells)
		self.ax.set_ylim(0, 1)
		self.ax.set_xlabel('Shell number')
		self.ax.set_ylabel('Mass fraction')
		self.ax.legend()

	#Function to adjust a single element in a single shell, updating the plot
	def edit_datapoint(self, symbol, shell_number, mass_fraction):
		self.data[symbol].profile[shell_number - 1] = mass_fraction
		self.lines[symbol].set_ydata(self.data[symbol].profile)
		self.fig.canvas.draw_idle()

	#Slider for a single element in a single shell
	def elem_slider(self, symbol, shell_number):
		interact(self.edit_datapoint, symbol = fixed(symbol),\
			shell_number = fixed(shell_number),\
			mass_fraction = FloatSlider(min=0.0, max=1.0, step=self.step, description = symbol + ' ' + str(shell_number), value = self.data[symbol].profile[shell_number - 1]))

	#Function to call sliders for all the shells for the requested element
	def slider(self, symbol):
		for j in range(0, self.shells):
			self.elem_slider(symbol.lower(), j+1)

	#Function to export the model as a file ready for use in TARDIS
	def export_model(self, filename):
		#Arranging the data to write
		lines = []
		for k in range(self.shells):
			lines.append([])
			for h in range(len(self.symbol_array)):
				lines[k].append(self.data[self.symbol_array[h]].profile[k])

		#Writing the data to file
		with open(filename, 'w') as outfile:
			for e in capitalise_array(self.symbol_array):
				outfile.write(e + ' ')
			outfile.write('\n\n')
			for i in lines:
				for j in i:
					outfile.write(str(j) + ' ')
				outfile.write('\n')

#Function to import a .dat abundance file as an instance of the model class, filling in the corresponding element_profiles
def import_model(filename):	
	with open(filename, 'r') as infile:
		infile_lines = infile.readlines()

	#Take roll call of the elements, in lower case, with the newline character removed
	elements = lower(clean_newline(infile_lines[0].split(sep = ' ')))
	
	#Extracting the numerical data from the file
	shell_comps = []
	for j in infile_lines[1:]:
		if len(j) == 1:
			continue
		else:
			shell_comps.append(clean_newline(j.split(sep = ' ')))

	#Arranging this data to create a 'model' instance
	new_profiles = []
	for k in elements:
		new_profiles.append(np.zeros(len(shell_comps)))

	for i in range(len(shell_comps)):
		for j in range(len(shell_comps[i])):
			new_profiles[j][i] = shell_comps[i][j]

	#Creating the model, initialised to zero for all elements in all shells
	out_model = model(elements, len(shell_comps))
	
	#Rewriting the element profiles with the imported data
	for i in range(len(elements)):
		out_model.data[elements[i]].change_profile(new_profiles[i])

	return out_model
