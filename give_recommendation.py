from flask import Flask, render_template,  request
import numpy as np 


app = Flask(__name__,static_url_path = "", static_folder = "static")









# features_list = np.load("data/resnet_features_list.npy")


# query_n = int(sys.argv[1])
# K = 6
# subtracted_list = np.subtract(features_list , features_list[query_n])
# distance_list = np.sum(np.abs(subtracted_list)**2,axis=-1)**(1./2)
# top_k_ids = np.argsort(distance_list)[:K]

# print(names_list[top_k_ids])



@app.route('/')
def load_data():
	N_init = 10
	names_list = np.load("data/names_list.npy")
	init_list = np.random.choice(len(names_list), N_init)
	init_names = []
	input_dir = 'dress_images/'
	
	for i in init_list:
		init_names.append(input_dir + names_list[i])

	return render_template('index.html',init_names = init_names)


@app.route('/handle_data', methods=['POST'])
def handle_data():
	im_chosen = request.form['chosen_image']
	im_chosen = im_chosen.split("/")[1]
	names_list = np.load("data/names_list.npy")
	features_list = np.load("data/resnet_features_list.npy")
	input_dir = 'dress_images/'
	query_n = np.where(names_list == im_chosen)
	K = 6
	subtracted_list = np.subtract(features_list , features_list[query_n])
	distance_list = np.sum(np.abs(subtracted_list)**2,axis=-1)**(1./2)
	top_k_ids = np.argsort(distance_list)[:K]
	selected_names = []
	for i in top_k_ids:
		selected_names.append(input_dir + names_list[i])

	return render_template('selection.html',selected_names = selected_names)



if __name__ == '__main__':

	app.run(debug = True)
