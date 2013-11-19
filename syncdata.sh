#!/bin/sh

rsync -rltvz wisteria.ki.tu-berlin.de:~/master/plume/.fridge/* ~/Documents/programming/uni/master/plume/.fridge/
rsync -rltvz wisteria.ki.tu-berlin.de:~/master/plume/Data/ ~/Documents/programming/uni/master/plume/Data/
rsync -rltvz wisteria.ki.tu-berlin.de:~/master/noise/plume/Data/ ~/Documents/programming/uni/master/plume/Data/
