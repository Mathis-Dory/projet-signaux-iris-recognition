# projet-signaux-iris-recognition
Projet dans le cadre du cours de traitement de signaux de 3TI à l'EPHEC

## Projet écrit en MatLab

Ce projet n'est pas fonctionnel, il n'a pas été achevé et est limité à la détection de la pupille et de l'iris.


## Projet écrit en Python

La version Python du projet est fonctionnelle. Cependant, il peut parfois détecter des faux négatifs.
Une image floue ou avec trop peu de contrastes dans l'iris ou avec une iris dont le contour passe hors du cadre de l'image provoque un faux négatif.

Il faut s'assurer que les images soient au format BMP et que les deux images concernées aient la même strcuture au niveau de l'array de pixels.

### Lancement du programme

Tapez la commande _python3 iris_recognition.py_ dans un terminal puis, selectionnez deux images.
