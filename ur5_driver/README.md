To install urx packages run the following in a bash terminal:

`pip install math3d`

`sed -i 's/import collections/import collections.abc as collections/' ~/.local/lib/python3.10/site-packages/math3d/utils.py`



`pip install urx`

`sed -i '216 s/./#&/' ~/.local/lib/python3.10/site-packages/urx/urrobot.py`
`sed -i '217 s/./#&/' ~/.local/lib/python3.10/site-packages/urx/urrobot.py`

``sed -i '252 s/./#&/' ~/.local/lib/python3.10/site-packages/urx/urrobot.py`

`sed -i '252s/.*/        try:\n            self.wait()\n        except:\n            self.close()/' ~/.local/lib/python3.10/site-packages/urx/ursecmon.py`