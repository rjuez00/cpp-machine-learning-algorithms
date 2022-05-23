#include "../includes/neurona.hpp"
#include <fstream>

Neurona::Neurona(float umbral, int tipo) {
    this->umbral = umbral;
    this->tipo = tipo;
    this->valor_entrada = 0.0;
    this->valor_salida = 0.0;
}

Neurona::~Neurona() {}

void Neurona::Inicializar(float x) {
    this->valor_entrada = x;
    //this->valor_salida = 0.0;
}

void Neurona::Conectar(Neurona *neurona, float peso) {
    this->conexiones.push_back(Conexion(peso, neurona));
}

void Neurona::Disparar() {

    if (tipo == Directa) {
        this->valor_salida = this->valor_entrada;
    } else if (tipo == Sesgo) {
        this->valor_salida = 1.0;
    } else if (tipo == McCulloch) {
        this->valor_salida = this->valor_entrada >= this->umbral ? 1.0 : 0.0;
    } else if (tipo == Adaline) {
        this->valor_salida = this->valor_entrada >= 0 ? 1.0 : -1.0;
    } else if (tipo == Perceptron) {
        if (this->valor_entrada > this->umbral) {
            this->valor_salida = 1.0;
        } else if (this->valor_entrada < -this->umbral) {
            this->valor_salida = -1.0;
        } else {
            this->valor_salida = 0.0;
        }
    } else {
        // Añadir mas tipos de neuronas
    }

    // TODO: Asegurarse de que se hace aqui
    //this->valor_entrada = 0.0;
}

void Neurona::Propagar() {
    for (auto & conexion: this->conexiones) {
        conexion.valor = this->valor_salida;
        conexion.Propagar();
    }
}
