# auto-attendence-system-AI
#### 1. Setup backend

Install all dependencies using conda package manager
> Note: This will install the dependencies listed in `environments.yml` file
```sh
$ cd backend
$ conda env create -f environment.yml
```
Now you can activate this environment using the following command
> Note: You can run the app only if this environment is activated
```sh
$ conda activate attendance-system
```
now rename .env.example to .env
## Usage

#### A. Using CLI
Follow these steps to run the app in command line interface mode
* Activate the `attendance-system` conda environment
* Launch `run_cli.py` from the backend directory
```sh
$ cd backend
$ conda activate attendance-system
$ python run_cli.py
```
#### B. Web backend server
Follow these steps to run the app in web server mode
* Activate the `attendance-system` conda environment
* Launch `web_app.py` from the backend directory
```sh
$ cd backend
$ conda activate attendance-system
$ python web_app.py
```
#### C. Setup FrontEnd

Install all front-end dependencies follow this command
> Note: cd front-end 
```sh
$ cd front-end
```
* now rename env.example to .env

```sh
$ npm install
$ npm run dev
```