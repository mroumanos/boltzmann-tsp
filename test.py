import requests

if __name__ == '__main__':

    data = {
        'distances' : [
            [0,10,20,5,18],
            [0,0,15,32,10],
            [0,0,0,25,16],
            [0,0,0,0,35],
            [0,0,0,0,0]
        ],
        'T' : 5000,
        'h_charge' : 0.5,
        'b_charge' : -0.2
    }

    res = requests.post('http://localhost:5000',json=data)

    with open('output.csv','w') as csv:
        for r in res.text:
            csv.write(r)
