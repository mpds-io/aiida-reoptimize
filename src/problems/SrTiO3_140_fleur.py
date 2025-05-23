from aiida.common.exceptions import NotExistent
from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict, Float, List, QueryBuilder, StructureData, load_node
from aiida.orm.nodes.data.code import Code
from aiida_fleur.workflows.relax import FleurRelaxWorkChain
from ase import Atoms


class ObjectiveFunctionWorkChain(WorkChain):
    """WorkChain for DFT-based energy calculation using FLEUR code.

    Implements structural relaxation workflow with error handling and output parsing.
    Follows AiiDA data provenance requirements.
    """  # noqa: E501

    @classmethod
    def define(cls, spec):
        """Define input/output ports and workflow steps.

        Args:
            spec: WorkChain specification object
        """
        super().define(spec)
        spec.input(
            "x",
            valid_type=List,
            help="Structure parameters [a, c] for unit cell",
        )
        spec.outline(
            cls.run_fleur_relax,  # Structural relaxation step
            cls.finalize_energy,  # Energy extraction step
        )
        spec.output(
            "energy",
            valid_type=Float,
            help="Total energy from DFT calculation",
        )

    def find_nodes(self, fleur_node_label, inpgen_node_label):
        qb = QueryBuilder()
        qb.append(Code, filters={"label": {"in": [fleur_node_label, inpgen_node_label]}})  # noqa: E501
        nodes = qb.all()
        
        if not nodes:
            raise ValueError("No Fleur or inpgen codes found in the database")
        
        data = {}
        for node in nodes:
            data[node[0].label] = node[0].pk
        return data

    def run_fleur_relax(self):
        fleur_node_label, inpgen_node_label = "fleur", "inpgen"
        nodes = self.find_nodes(fleur_node_label, inpgen_node_label)
        required_codes = [fleur_node_label, inpgen_node_label]
        for code_label in required_codes:
            if code_label not in nodes:
                raise KeyError(f"Missing required code: {code_label}")
            
            try:
                fleur_code = load_node(nodes[fleur_node_label])
                inpgen_code = load_node(nodes[inpgen_node_label])
            except NotExistent as e:
                raise RuntimeError(f"Failed to load code node: {e}") from e

        # Extract unit cell parameters
        a = self.inputs.x.get_list()[0]
        c = self.inputs.x.get_list()[1]
        print(f"Current cell parameters: a={a}, c={c}")

        # Create unit cell matrix
        cell = [
            [-0.5 * a, 0.5 * a, 0.5 * c],
            [0.5 * a, -0.5 * a, 0.5 * c],
            [0.5 * a, 0.5 * a, -0.5 * c],
        ]

        # Atomic positions (fractional coordinates)
        scaled_positions = [
            (0.75, 0.25, 0.5),  # Sr
            (0.25, 0.75, 0.5),  # Sr
            (0.0, 0.0, 0.0),  # Ti
            (0.5, 0.5, 0.0),  # Ti
            (0.25894899, 0.24105101, 0.5),  # O
            (0.74105101, 0.75894899, 0.5),  # O
            (0.24105101, 0.74105101, 0.98210202),  # O
            (0.75894899, 0.25894899, 0.01789798),  # O
            (0.25, 0.25, 0.0),  # O
            (0.75, 0.75, 0.0),  # O
        ]

        # Create ASE Atoms object for structure
        symbols = ["Sr", "Sr", "Ti", "Ti", "O", "O", "O", "O", "O", "O"]
        ase_structure = Atoms(
            symbols=symbols,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=True,
        )

        # Convert to AiiDA StructureData
        structure = StructureData(ase=ase_structure)

        # Relaxation parameters
        wf_relax = Dict(
            dict={
                "film_distance_relaxation": False,
                "force_criterion": 0.049,  # Convergence threshold (eV/Å)
                "relax_iter": 10,  # Maximum relaxation steps
            }
        )

        # SCF calculation parameters
        wf_relax_scf = Dict(
            dict={
                "fleur_runmax": 10,  # Max SCF cycles
                "itmax_per_run": 100,  # Max iterations per SCF
                "energy_converged": 0.0001,  # Energy convergence (eV)
                "mode": "energy",  # Convergence criterion
                "force_dict": {  # Force mixing parameters
                    "qfix": 2,
                    "forcealpha": 0.5,
                    "forcemix": "straight",
                },
            }
        )

        # Computational resource configuration
        options_relax = Dict(
            dict={
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 4,
                },
                "queue_name": "devel",
                "max_wallclock_seconds": 10 * 60 * 6000,  # 10 hours
            }
        )

        # Submit FleurRelaxWorkChain
        future = self.submit(
            FleurRelaxWorkChain,
            scf={
                "wf_parameters": wf_relax_scf,
                "options": options_relax,
                "inpgen": inpgen_code,
                "fleur": fleur_code,
                "structure": structure,
            },
            wf_parameters=wf_relax,
        )
        return ToContext(energy_future=future)  # Track calculation status

    def finalize_energy(self):
        """
        Extract and validate energy from completed calculation.
        """
        energy_value = None

        if self.ctx.energy_future.is_finished_ok:
            try:
                # Retrieve energy from work chain outputs
                output_para = (
                    self.ctx.energy_future.outputs.output_relax_wc_para
                )
                total_energy = output_para.get_dict().get("energy")

                if total_energy is not None:
                    energy_value = total_energy
                else:
                    self.report("Warning: Energy value missing in outputs")

            except NotExistent:
                self.report("Error: Output parameters node not found")
            except Exception as e:
                self.report(f"Error parsing results: {str(e)}")

        # Store energy value in AiiDA database
        energy_node = Float(energy_value).store()
        self.out("energy", energy_node)
