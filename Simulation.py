from Community import Community
from Node import Node


class Simulation:
    @staticmethod
    def main():
        # Initialize communities and nodes
        nodes = []
        for c_i in range(1, 10):
            c = Community(0.0, 0.0)
            for n_i in range(1, 10):
                nodes.append(Node(len(nodes), c))

        # Run simulation
        for t in range(1, 100):
            print("t: " + str(t))
            for n in nodes:
                n.move()
                n.learn()

            for n1 in nodes:
                for n2 in nodes:
                    if n1 != n2 and n1.sees(n2):
                        print(str(n1.identifier) + " sees " + str(n2.identifier))
                        n1.share(n2)

        for n in nodes:
            n.evaluate()


if __name__ == '__main__':
    Simulation.main()
