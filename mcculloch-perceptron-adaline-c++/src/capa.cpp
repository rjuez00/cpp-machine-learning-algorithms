#include "../includes/capa.hpp"
#include <random>


Capa::Capa() {}

Capa::~Capa() {}

void Capa::Inicializar() {
    for (auto neurona: neuronas) {
        // TODO: Que float pasar
        neurona->Inicializar(0.0);
    }
}

void Capa::Anadir(Neurona *neurona) {
    this->neuronas.push_back(neurona);
}

void Capa::Conectar(Capa *Capa, float peso_min, float peso_max) {
    for (auto & neuronaEnOtraCapa: Capa->neuronas) {
        this->Conectar(neuronaEnOtraCapa, peso_min, peso_max);
    }
}

void Capa::Conectar(Neurona *neurona, float peso_min, float peso_max) {
    for (auto neuronaEnCapa: this->neuronas) {
        float peso_rand = peso_min + static_cast <float> (rand()) / ( static_cast <float> (RAND_MAX/(peso_max-peso_min)));
        neuronaEnCapa->Conectar(neurona, peso_rand);
    }
}

void Capa::Disparar() {
    for (auto neuronaEnCapa: this->neuronas) {
        neuronaEnCapa->Disparar();
    }
}

void Capa::Propagar() {
    for (auto neuronaEnCapa: this->neuronas) {
        neuronaEnCapa->Propagar();
    }
}
