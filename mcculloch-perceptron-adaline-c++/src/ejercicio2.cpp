#include "../includes/redneuronal.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


int main(int argc, char** argv) {

    if(argc < 2){
        std::cout << "Haga 'make ayuda_mp'\n";
        exit(1);
    }

    // Creacion de la red
    float theta = 2.0;
    RedNeuronal red;
    Capa entradaC, salidaC, ocultaC;
    Neurona X1(NO_UMBRAL, Neurona::Directa),
            X2(NO_UMBRAL, Neurona::Directa),
            Z1(theta, Neurona::McCulloch),
            Z2(theta, Neurona::McCulloch),
            Y1(theta, Neurona::McCulloch),
            Y2(theta, Neurona::McCulloch);

    X1.Conectar(&Y1, 2.0);
    X2.Conectar(&Z1, -1.0);
    X2.Conectar(&Z2, 2.0);
    X2.Conectar(&Y2, 1.0);
    Z1.Conectar(&Y1, 2.0);
    Z2.Conectar(&Z1, 2.0);
    Z2.Conectar(&Y2, 1.0);

    entradaC.Anadir(&X1);
    entradaC.Anadir(&X2);
    ocultaC.Anadir(&Z1);
    ocultaC.Anadir(&Z2);
    salidaC.Anadir(&Y1);
    salidaC.Anadir(&Y2);

    red.Anadir(&entradaC);
    red.Anadir(&ocultaC);
    red.Anadir(&salidaC);

    // LECTURA INPUT
    std::ifstream input;
    input.open(argv[1]);
    if(!input){
        std::cout << "El archivo " << argv[1] << " no existe";
        exit(1);
    }

    // EJECUCION
    std::ofstream output;
    bool no_output_file = false;
    if (argc < 3){
       no_output_file = true;
    } else {
        output.open(argv[2], std::ios::trunc);
        if(!output){
            std::cout << "El archivo " << argv[1] << " no existe";
            exit(1);
        }
    }

    output << "x1\tx2\tz1\tz2\ty1\ty2\n";

    // Bucle de lectura de inputs e impresión de valores de las neuronas
    float x1, x2;
    std::stringstream buffer;
    while (input >> x1 >> x2) {
        X1.valor_entrada = x1;
        X2.valor_entrada = x2;
        red.Disparar();
        red.Inicializar();
        red.Propagar();

        buffer.str(std::string());
        buffer << X1.valor_salida << "\t" << X2.valor_salida << "\t" << Z1.valor_salida << "\t" << Z2.valor_salida << "\t" << Y1.valor_salida << "\t" << Y2.valor_salida << "\n";
        if (no_output_file) std::cout << buffer.str();
        else                   output << buffer.str();
        
    }

    // Continua hasta que se hayan propagado todas las entradas
    while (   Z1.valor_salida != 0.0 
           || Z2.valor_salida != 0.0 
           || Y1.valor_salida != 0.0 
           || Y2.valor_salida != 0.0) {

        red.Disparar();
        red.Propagar();

        buffer.str(std::string());
        buffer << X1.valor_salida << "\t" << X2.valor_salida << "\t" << Z1.valor_salida << "\t" << Z2.valor_salida << "\t" << Y1.valor_salida << "\t" << Y2.valor_salida << "\n";
        if (no_output_file) std::cout << buffer.str();
        else                   output << buffer.str();
    }

}
