import machine
import math, sys
import numpy as np
from flask import Flask, request, Response
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@app.route('/', methods=['POST'])
def boltzmann_handler(event=None, context=None):
    print('in boltzmann_handler')
    
    data = request.get_json()
    distances = np.matrix(data['distances'])
    # making it symmetric for ease of use
    distances = distances + distances.T
    temp_start = int(data['T'])
    h_charge = float(data['h_charge'])
    b_charge = float(data['b_charge'])

    print('distances = %s' % distances)
    print('temp_start = %s' % temp_start)
    print('h_charge = %s' % h_charge)
    print('b_charge = %s' % b_charge)

    # creates the boltzmann machine
    b = machine.boltzmann(hamiltonian_error_charge=h_charge, bias_charge=b_charge)
    b.create_network(distances)

    # creates the output generator
    def generate(temp):
        for row in machine.anneal(b,T=temp,schedule=lambda t: math.log10(t) if t > 100 else 0.1):
            yield ','.join(row) + '\n'

    # streams the generator output in its response
    return Response(generate(temp_start), mimetype='text/csv')

if __name__ == '__main__':

    app.run(debug=True)