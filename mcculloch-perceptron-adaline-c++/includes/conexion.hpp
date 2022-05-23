#ifndef CONEXION_H
#define CONEXION_H


class Neurona;

class Conexion {
public:
    Conexion(float peso, Neurona *neurona);
    virtual ~Conexion();
    void Propagar();

public:
    float peso;
    float peso_anterior;
    float valor;
    Neurona *neurona;
};

#endif /* CONEXION_H */
