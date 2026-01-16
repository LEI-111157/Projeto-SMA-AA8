from sim.motor_de_simulacao import MotorDeSimulacao


if __name__ == "__main__":
    motor = MotorDeSimulacao.cria("params/foraging_novelty_test.json")
    motor.executa()






