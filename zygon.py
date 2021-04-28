"""
▄▀█ █▄▄ █░█ █▄░█ █▀▄ ▄▀█ █▄░█ █▀▀ █▀▀   █▀▀ █▀▄ █ ▀█▀ █▀█ █▀█
█▀█ █▄█ █▄█ █░▀█ █▄▀ █▀█ █░▀█ █▄▄ ██▄   ██▄ █▄▀ █ ░█░ █▄█ █▀▄

Luke Harvey - 28/04/2021
Trinity College Dublin

Importing this script into a jupyter notebook with ' %matplotlib notebook ' set will allow these functions to run.
An accompanying notebook can be found the GitHub repo.

08/04/2021 - Adding functionality to build a full abundance model for the supernova and then take slices as the profiles at different epochs as the photosphere receeds.

28/04/2021 - Rewrote the export_csvy function that I accidentally deleted. Cleaning up the Git structure before I add documentation comments for sphinx.
"""

from ipywidgets import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

dimensions = (7, 5)
full_dimensions = (8, 5)

root_path = os.path.dirname(__file__)

#Used as the resolution to create the list from which line colours will be extracted
colour_step = 5
#Matplotlib markers and linestyles to cycle through for each of the different elements
markers = ['o']
linestyles = ['-']
grey = '#FFFFFF'

def where(array, value):
	index = 0
	for i in range(len(array)):
		if array[i] == value:
			index = i 
			break
	return index

def round_sf(num, sf):
	exp = np.floor(np.log10(num))
	
	tenexp = 10**exp

	if exp < 0:
		return np.round(num/tenexp, sf) * tenexp
	else:
		return np.round(num/tenexp, sf - 1) * tenexp

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

#Class to initalise and hold the density profile information
class density_profile():
	def __init__(self, shells, velocities, init_density):
		self.x = np.arange(1, shells+1, 1)
		self.vels = np.linspace(velocities[0], velocities[1], shells + 1)
		self.profile = np.zeros_like(self.vels)
		self.init_density = init_density

		for i in range(len(self.profile)):
			self.profile[i] = init_density

	def exp_original(self, x, a, rho0):
		factor = np.exp(-(x[0]/10000)*a)
		return rho0*np.exp(-(x/10000)*a) / factor

	def exp(self, velocities, velocity_0, rho0):
		return rho0 * np.exp(-(velocities / velocity_0))

	def power_original(self, x, a, rho0):
		return rho0*(x/10000)**(-a)

	def power(self, velocities, velocity_0, rho0, a):
		return rho0 * np.power((velocities / velocity_0), a)

def plot_models(models, legend = False, titles = None):
	rows = int(np.ceil(len(models)/2))
	fig, axs = plt.subplots(rows, 2, figsize = (18, 8*rows))

	for x in range(len(models)):
		i = int(np.floor(x/2))
		j = 0
		if x%2 != 0:
			j = 1

		for element_index in range(len(models[x].symbol_array)):
			axs[i, j].plot(models[x].x, models[x].abundance_data[models[x].symbol_array[element_index]].profile, color = models[x].colours[element_index], marker = 'o', label = models[x].symbol_array[element_index])

	for x in range(len(models)):
		i = int(np.floor(x/2))
		j = 0
		if x%2 != 0:
			j = 1

		axs[i, j].set_ylim(0, 1)

		if titles == None:
			axs[i, j].set_title(str(x + 1))
		else:
			axs[i, j].set_title(titles[x])


	if legend:
		axs[0, 0].legend()
	plt.show()

class model():

	#█ █▄░█ █ ▀█▀ █ ▄▀█ █░░ █ █▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
	#█ █░▀█ █ ░█░ █ █▀█ █▄▄ █ ▄█ █▀█ ░█░ █ █▄█ █░▀█
	
	def __init__(self, symbol_array = ['c', 'o', 'na', 'mg', 'si', 's', 'ca', 'ti', 'cr', 'fe', 'ni'], shells = 30, velocities = (0, 25000), init_density = 1e-13):
		#Establishing the number of shells as well as an array 'x' for plotting
		self.shells = shells
		self.x = np.arange(1, shells+1, 1)
		self.start_vel = velocities[0]
		self.stop_vel = velocities[1]

		#All the 'element_profile' instances will be stored in the dictionary 'data' each with their corresponding symbol
		self.abundance_data = {}
		self.density_data = density_profile(shells, velocities, init_density)
		
		#Storing the list of requested elements as a parameter
		self.symbol_array = []
		for i in symbol_array:
			self.symbol_array.append(i.lower())

		#Calling the 'write_new_element' function to populate the dictionary 'data'
		for i in self.symbol_array:
			self.write_new_element(i)

		self.tick_locs = []

		for i in range(shells):
			self.tick_locs.append(i + 1)



		self.init_colours()

	#Add element_profile classes to the dictionary 'data'
	def write_new_element(self, symbol):

		self.abundance_data[symbol] = element_profile(symbol, self.shells)

	def init_colours(self):
		#Sampling the 'rainbow.rainbow' array to get an even spread of colours for plotting
		self.colours = []
		self.step = 0.02

		if len(self.symbol_array) != 0:
			for i in np.arange(0, len(rainbow.rainbow)-1, int(np.ceil(len(rainbow.rainbow)/len(self.symbol_array)))):
				self.colours.append(rainbow.rainbow[i])

	#█▀█ █░░ █▀█ ▀█▀ █▀
	#█▀▀ █▄▄ █▄█ ░█░ ▄█
	
	def plot(self):
		self.combined = True

		self.combined_fig = plt.figure(num = 'Profiles', figsize = full_dimensions)
		self.abundance_ax = self.combined_fig.add_subplot(1, 1, 1)
		self.density_ax = self.combined_fig.add_subplot(1, 1, 1, frame_on = False)

		#Each line will be stored in the dictionary 'lines' - similar to the 'data' dictionary they can be called by their corresponding symbol
		self.lines = {}
		for i in range(len(self.symbol_array)):
			self.lines[self.symbol_array[i]], = self.abundance_ax.plot(self.x, self.abundance_data[self.symbol_array[i]].profile,\
				marker = markers[i%len(markers)], label = self.symbol_array[i], color = self.colours[i],\
				linestyle = linestyles[i%len(linestyles)])

		
		self.density_ax.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)
		self.density_ax.yaxis.set_label_position('right')
		miny, maxy = np.amin(self.density_data.profile), np.amax(self.density_data.profile)
		#self.density_ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
		if miny == maxy:
			self.density_ax.set_ylim(0, 2*self.density_data.init_density)
		else:
			self.density_ax.set_ylim(miny, maxy)
		self.density_ax.yaxis.set_label_coords(1.11, 0.5)
		self.density_ax.set_xlabel(r"Veloctiy ($km$ $s^{-1}$)")
		self.density_ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
		self.density_ax.xaxis.set_label_position('top')
		self.density_ax.set_xlim(self.start_vel, self.stop_vel)
		#self.density_ax.yaxis.offsetText.set_visible(False)
		self.density_ax.yaxis.set_offset_position('right')
		self.density_ax.set_ylabel(r"Density ($g$ $cm^{-3}$)", rotation = 270)

		#Each line will be stored in the dictionary 'lines' - similar to the 'data' dictionary they can be called by their corresponding symbol
		self.density_line, = self.density_ax.plot(self.density_data.vels, self.density_data.profile, marker = 'X',\
			label = r"$\rho$ profile", color = grey, linestyle = '-.')

		#Plot housekeeping
		self.abundance_ax.set_xlim(0.5, self.shells+0.5)
		self.abundance_ax.set_xlabel('Shell number')
		self.abundance_ax.set_ylim(0, 1)
		self.abundance_ax.set_xticks(self.tick_locs)
		self.abundance_ax.set_ylabel('Mass fraction')

		handles, labels = [(a + b) for a, b in zip(self.abundance_ax.get_legend_handles_labels(), self.density_ax.get_legend_handles_labels())]

		#self.combined_fig.legend(handles, labels)
		self.abundance_ax.legend(handles, labels)
		plt.tight_layout()
		plt.show()

	#▄▀█ █▄▄ █░█ █▄░█ █▀▄ ▄▀█ █▄░█ █▀▀ █▀▀   █▀ █░░ █ █▀▄ █▀▀ █▀█ █▀
	#█▀█ █▄█ █▄█ █░▀█ █▄▀ █▀█ █░▀█ █▄▄ ██▄   ▄█ █▄▄ █ █▄▀ ██▄ █▀▄ ▄█
	
	#Function to adjust a single element in a single shell, updating the plot
	def edit_abundance_datapoint(self, symbol, shell_number, mass_fraction):
		self.abundance_data[symbol].profile[shell_number - 1] = mass_fraction
		self.lines[symbol].set_ydata(self.abundance_data[symbol].profile)
		self.combined_fig.canvas.draw_idle()

	#Slider for a single element in a single shell
	def elem_slider(self, symbol, shell_number):
		interact(self.edit_abundance_datapoint, symbol = fixed(symbol),\
			shell_number = fixed(shell_number),\
			mass_fraction = FloatSlider(min=0.0, max=1.0, step=self.step, description = symbol + ' ' + str(shell_number), value = self.abundance_data[symbol].profile[shell_number - 1]))

	#Function to call sliders for all the shells for the requested element
	def abundance_slider(self, symbol):
		for j in range(0, self.shells):
			self.elem_slider(symbol.lower(), j+1)

	#█▀▄ █▀▀ █▄░█ █▀ █ ▀█▀ █▄█   █▀ █░░ █ █▀▄ █▀▀ █▀█ █▀
	#█▄▀ ██▄ █░▀█ ▄█ █ ░█░ ░█░   ▄█ █▄▄ █ █▄▀ ██▄ █▀▄ ▄█

	#Function to adjust a single element in a single shell, updating the plot
	def edit_density_datapoint(self, velocity, density):
		index = where(self.density_data.vels, velocity)
		self.density_data.profile[index] = 10**density
		self.density_line.set_ydata(self.density_data.profile)
		#self.density_ax.set_ylim(np.amin(self.density_data.profile), np.amax(self.density_data.profile))

		self.combined_fig.canvas.draw_idle()

	#Slider for a single element in a single shell
	def density_point_slider(self, velocity, index):
		index = where(self.density_data.vels, velocity)
		current = self.density_data.profile[index]
		log_current = np.log10(current)		

		interact(self.edit_density_datapoint, velocity = fixed(velocity),\
			density = FloatSlider(min=log_current-1, max=log_current+1, step=0.001, description = str(index) + "/" + str(index+1), value = log_current))

	#Function to call sliders for all the shells for the requested element
	def density_shell_slider(self):
		self.density_ax.set_ylim(np.amin(self.density_data.profile), np.amax(self.density_data.profile))
		for j in range(len(self.density_data.vels)):
			self.density_point_slider(self.density_data.vels[j], j)

	def edit_density_curve_exp(self, vel_0, rho0_index):
		self.density_data.profile = self.density_data.exp(self.density_data.vels, vel_0, 10**rho0_index)
		self.density_line.set_ydata(self.density_data.profile)
		self.density_ax.set_ylim(np.amin(self.density_data.profile), np.amax(self.density_data.profile))

		self.combined_fig.canvas.draw_idle()

	def edit_density_curve_power(self, vel_0, rho0_index, a):
		#self.density_data.profile = self.density_data.power(self.density_data.vels, a, 10**rho0_index)
		self.density_data.profile = self.density_data.power(self.density_data.vels, vel_0, 10**rho0_index, a)
		self.density_line.set_ydata(self.density_data.profile)
		self.density_ax.set_ylim(np.amin(self.density_data.profile), np.amax(self.density_data.profile))


		self.combined_fig.canvas.draw_idle()

	def density_slider(self, law):
		if law.lower() == 'exp' or law.lower() == 'exponential':
			#self.curve_law = True
			#interact(self.edit_density_curve_exp, a = FloatSlider(min=0.01, max=20.0, step=0.01), rho0_index = FloatSlider(min=-20, max=-12, step=0.1, description = 'rho0'))
			interact(self.edit_density_curve_exp, vel_0 = FloatSlider(min=0.1*self.start_vel, max=2*self.stop_vel, step=10, description = "v0"), rho0_index = FloatSlider(min=-14, max=-9, step=0.01, description = "rho0"))


		if law.lower() == 'power':
			#self.curve_law = True
			#interact(self.edit_density_curve_power, a = FloatSlider(min=0.1, max=20.0, step=0.01), rho0_index = FloatSlider(min=-20, max=-12, step=0.1, description = 'rho0'))
			interact(self.edit_density_curve_power, vel_0 = FloatSlider(min=0.1*self.start_vel, max=2*self.stop_vel, step=10, description = "v0"), rho0_index = FloatSlider(min=-20, max=-12, step=0.1, description = 'rho0'),\
				a = FloatSlider(min = -10, max = -0.1, step = 0.01))

	#█▀▀ ▀▄▀ █▀█ █▀█ █▀█ ▀█▀
	#██▄ █░█ █▀▀ █▄█ █▀▄ ░█░

	#Function to export the model as a file ready for use in TARDIS
	def export_abundance(self, filename = 'abundance_output.dat'):
		#Pass through and normalise each of the shells
		for n in range(self.shells):
			summation = 0
			for element in self.symbol_array:
				summation += self.abundance_data[element].profile[n]

			scale_factor = 1/summation
			for element in self.symbol_array:
				self.abundance_data[element].profile[n] = self.abundance_data[element].profile[n] * scale_factor

		#Arranging the data to write
		lines = []
		for k in range(self.shells):
			lines.append([])
			for h in range(len(self.symbol_array)):
				lines[k].append(self.abundance_data[self.symbol_array[h]].profile[k])

		capitalised_elements = self.symbol_array.copy()
		#Writing the data to file
		with open(filename, 'w') as outfile:
			outfile.write('Index ')
			for e in capitalise_array(capitalised_elements):
				outfile.write(e + ' ')
			outfile.write('\n\n')
			for i in range(len(lines)):
				outfile.write(str(i) + ' ')
				for j in lines[i]:
					outfile.write(str( '%0.4f' % j ) + ' ')
				outfile.write('\n')

	def export_density(self, filename = 'density_output.txt', day = 1):
		lines = []
		for k in range(self.shells+1):
			lines.append([])
			lines[k].append(k)
			lines[k].append(self.density_data.vels[k])
			lines[k].append(round_sf(self.density_data.profile[k], 4))

		with open(filename, 'w') as outfile:
			outfile.write(str(day) + ' day\n\n')

			for j in lines:
				for i in j:
					outfile.write(str(i) + ' ')
				outfile.write('\n')

	#Function to export the model as a file ready for use in TARDIS
	def export_abundance_slice(self, filename = 'abundance_slice_output.dat', shells = (1, 2), slice_shells = 11):
		shell_low, shell_high = shells
		old_shells = np.arange(1, self.shells + 1, 1)

		new_shells = np.linspace(shell_low, shell_high, slice_shells)

		new_abundance_data = {}

		for i in self.symbol_array:
			new_abundance_data[i] = element_profile(i, slice_shells)

			new_abundance_data[i].profile = np.interp(new_shells, old_shells, self.abundance_data[i].profile)

		#Pass through and normalise each of the shells
		for n in range(slice_shells):
			summation = 0
			for element in self.symbol_array:
				summation += new_abundance_data[element].profile[n]

			scale_factor = 1/summation
			for element in self.symbol_array:
				new_abundance_data[element].profile[n] = new_abundance_data[element].profile[n] * scale_factor

		#Arranging the data to write
		lines = []
		for k in range(slice_shells):
			lines.append([])
			for h in range(len(self.symbol_array)):
				lines[k].append(new_abundance_data[self.symbol_array[h]].profile[k])

		capitalised_elements = self.symbol_array.copy()
		#Writing the data to file
		with open(filename, 'w') as outfile:
			outfile.write('Index ')
			for e in capitalise_array(capitalised_elements):
				outfile.write(e + ' ')
			outfile.write('\n\n')
			for i in range(len(lines)):
				outfile.write(str(i) + ' ')
				for j in lines[i]:
					outfile.write(str( '%0.4f' % j ) + ' ')
				outfile.write('\n')

	def export_density_slice(self, filename = 'density_slice_output.txt', shells = (1, 2), slice_shells = 11, day = 1):
		shell_low, shell_high = shells
		old_shells = np.arange(0.5, self.shells + 0.6, 1)

		new_shells = np.linspace(shell_low, shell_high, slice_shells)

		new_density_data = np.interp(new_shells, old_shells, self.density_data.profile)
		new_velocity_data = np.interp(new_shells, old_shells, self.density_data.vels)

		lines = []
		lines.append(str(day) + ' day\n\n')
		for i in range(len(new_shells)):
			lines.append(str(i) + ' ' + str(new_velocity_data[i]) + ' ' + str(new_density_data[i]) + '\n')

		with open(filename, 'w') as outfile:
			for i in lines:
				outfile.write(i)

	def export_csvy(self, filename = 'csvy_output.csvy', day = 'INSERT_DAY', velocities = ('INSERT_INNER_VELOCITY', 'INSERT_OUTER_VELOCITY')):
		with open(root_path + "/csvy_template.txt", 'r') as file:
			lines = file.readlines()

		output = []
		for i in lines:
			if 'model_density_time_0:' in i:
				time_line = i.split(sep = ' ')
				time_line[1] = str(day)
				reconstructed = time_line[0] + ' ' + time_line[1] + ' ' + time_line[2]
				output.append(reconstructed)
			else:
				output.append(i)

		capitalised_elements = capitalise_array(self.symbol_array.copy())

		for symbol in capitalised_elements:
			output.append('    -  name: ' + symbol + '\n       desc: fractional ' + symbol + ' abundance\n')

		output.append('\n\nv_inner_boundary: ' + str(velocities[0]) + ' km/s\nv_outer_boundary: ' + str(velocities[1]) + ' km/s\n---\n')

		csv_header = 'velocity,density'

		for symbol in capitalised_elements:
			csv_header += ',' + symbol

		output.append(csv_header)

		for i in range(self.shells):
			csv_line = str(self.density_data.vels[i+1]) + ',' + str(self.density_data.profile[i+1])

			for symbol in self.symbol_array:
				csv_line += ',' + str(self.abundance_data[symbol].profile[i])

			csv_line += '\n'
			output.append(csv_line)

		with open(filename, 'w') as file:
			for i in output:
				file.write(i)

	#█ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀
	#█ █░▀░█ █▀▀ █▄█ █▀▄ ░█░

	#Function to import an abundance model
	def import_abundance(self, filename, index = True):
		with open(filename, 'r') as infile:
			infile_lines = infile.readlines()	

		if index:
			start = 1
		else:
			start = 0

		#Take roll call of the elements, in lower case, with the newline character removed
		elements = lower(clean_newline(infile_lines[0].split(sep = ' ')))
		elements = elements[start:]
		
		#Extracting the numerical data from the file
		shell_comps = []
		for j in infile_lines[2:]:
			if len(j) == 1:
				continue
			else:
				shell_comps.append( clean_newline(j.split(sep = ' '))[start:] )	

		#Arranging this data to create a 'model' instance
		new_profiles = []
		for k in elements:
			new_profiles.append(np.zeros(len(shell_comps)))	

		for i in range(len(shell_comps)):
			for j in range(len(shell_comps[i])):
				new_profiles[j][i] = shell_comps[i][j]

		"""
		if self.shells > len(shell_comps):
			print("WARNING: The abundance file " + filename + " has less shells than the model. This will cause interpolation of the model structure to match.")

		if self.shells < len(shell_comps):
			print("WARNING: The abundance file " + filename + " has more shells than the model. This will cause padding of the model structure to match.")
		"""

		self.shells = len(shell_comps)
		self.x = np.arange(1, self.shells+1, 1)
		self.symbol_array = elements
		self.abundance_data = {}

		self.tick_locs = []

		for i in range(self.shells):
			self.tick_locs.append(i + 1)

		for i in self.symbol_array:
			self.write_new_element(i)

		for i in range(len(elements)):
			self.abundance_data[elements[i]].change_profile(new_profiles[i])

		self.init_colours()

	def import_density(self, filename):
		with open(filename, 'r') as infile:
			infile_lines = infile.readlines()[2:]

		in_vel = []
		in_den = []

		for i in infile_lines:
			line = i.split(sep = ' ')
			in_vel.append(float(line[1]))
			in_den.append(float(line[2]))

		if self.shells > len(infile_lines) - 1:
			print("WARNING: The density file " + filename + " has less shells than the model. This will cause interpolation of the current abundance profile to match.")
			#INTERPOLATE CURRENT ABUNDANCE PROFILE
			interp_shells = np.linspace(np.amin(self.x), np.amax(self.x), len(infile_lines)-1)
			
			for i in self.symbol_array:
				self.abundance_data[i].profile = np.interp(interp_shells, self.x, self.abundance_data[i].profile)

			self.x = np.arange(1, len(infile_lines), 1)
			#INTERPOLATE CURRENT ABUNDANCE PROFILE

		if self.shells < len(infile_lines) - 1:
			print("WARNING: The density file " + filename + " has more shells than the model. This will cause padding of the current abundance profile to match.")
			#PAD CURRENT ABUNDANCE PROFILE
			for i in range(self.shells, len(infile_lines)-1):
				self.x = np.append(self.x, i+1)

				for h in self.symbol_array:
					self.abundance_data[h].profile = np.append(self.abundance_data[h].profile, 0)
			#PAD CURRENT ABUNDANCE PROFILE

		self.shells = len(infile_lines) - 1

		self.tick_locs = []
		for i in range(self.shells):
			self.tick_locs.append(i + 1)

		self.start_vel = in_vel[0]
		self.stop_vel = in_vel[len(in_vel) - 1]
		self.density_data.vels = np.array(in_vel)
		self.density_data.profile = np.array(in_den)

	def set_velocities(self, start, stop):
		self.start_vel = start 
		self.stop_vel = stop 
