# Flight-delays
Set up

Dependencies:
python 3.5
pip install pandas matplotlib tensorflow==2.0.0-rc0 SciPy numpy sklearn

Alternatively run this docker container, which also contains the repository:
docker run -it --rm jeremycollinsmpi/flight-data:latest

To run the script:

python main.py --mode train

or 

python main.py --mode evaluate


![alt text](https://github.com/JeremyCollinsMPI/Flight-delays/blob/master/dag1.png)
