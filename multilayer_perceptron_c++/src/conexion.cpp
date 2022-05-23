#include "../includes/conexion.hpp"
#include "../includes/neurona.hpp"


Conexion::Conexion(float peso, Neurona *neurona) {
    this->peso = peso;
    this->neurona = neurona;
    
    this->peso_anterior = -1; // TODO que poner
}

Conexion::~Conexion() {

}

void Conexion::Propagar() {
    this->neurona->valor_entrada += this->peso * this->valor;
    this->valor = 0;
}
