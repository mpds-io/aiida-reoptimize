from aiida.common.exceptions import NotExistent
from aiida.engine import ToContext, WorkChain
from aiida.orm import Dict, Float, List, StructureData, load_node
from aiida_fleur.workflows.relax import FleurRelaxWorkChain
from ase import Atoms

from aiida.orm import QueryBuilder
from aiida.orm.nodes.data.code import Code


class ObjectiveFunctionWorkChain(WorkChain):
    """WorkChain for DFT-based energy calculation using FLEUR code.
    
    Implements structural relaxation workflow with error handling and output parsing.
    Follows AiiDA data provenance requirements.
    """
    
    @classmethod
    def define(cls, spec):
        """Define input/output ports and workflow steps.
        
        Args:
            spec: WorkChain specification object
        """
        super().define(spec)
        spec.input('x', valid_type=List, help='Structure parameters [a, c] for unit cell')
        spec.outline(
            cls.run_fleur_relax,    # Structural relaxation step
            cls.finalize_energy     # Energy extraction step
        )
        spec.output('energy', valid_type=Float, help='Total energy from DFT calculation')

    def find_nodes(self):
        qb = QueryBuilder()
        qb.append(Code, filters={'label': {'in': ['fleur', 'inpgen']}})
        nodes = qb.all()

        data = {}
        for node in nodes:
            data[node[0].label] = node[0].pk
        return data

    def run_fleur_relax(self):
        """Set up and execute FleurRelaxWorkChain for structural optimization.
        
        Configuration details:
        - Uses pre-configured FLEUR and inpgen codes (PK 4060/4061)
        - Creates SrTiO3-like structure with ASE
        - Sets relaxation parameters for metal-organic interface
        """
        # Load pre-configured code nodes
        nodes = self.find_nodes()
        FLEUR_PK = nodes['fleur']  # FLEUR code persistent identifier
        INPGEN_PK = nodes['inpgen']  # inpgen code persistent identifier
        fleur_code = load_node(FLEUR_PK)
        inpgen_code = load_node(INPGEN_PK)

        # Extract unit cell parameters
        a = self.inputs.x.get_list()[0]
        c = self.inputs.x.get_list()[1]
        print(f'Current cell parameters: a={a}, c={c}')
        
        # Create unit cell matrix
        cell = [
            [-0.5 * a, 0.5 * a, 0.5 * c],
            [0.5 * a, -0.5 * a, 0.5 * c],
            [0.5 * a, 0.5 * a, -0.5 * c]
        ]

        # Atomic positions (fractional coordinates)
        scaled_positions = [
            (0.75, 0.25, 0.5),         # Sr
            (0.25, 0.75, 0.5),         # Sr
            (0.0, 0.0, 0.0),           # Ti
            (0.5, 0.5, 0.0),           # Ti
            (0.25894899, 0.24105101, 0.5),  # O
            (0.74105101, 0.75894899, 0.5),  # O
            (0.24105101, 0.74105101, 0.98210202),  # O
            (0.75894899, 0.25894899, 0.01789798),  # O
            (0.25, 0.25, 0.0),         # O
            (0.75, 0.75, 0.0)          # O
        ]

        # Create ASE Atoms object for structure
        symbols = ['Sr', 'Sr', 'Ti', 'Ti', 'O', 'O', 'O', 'O', 'O', 'O']
        ase_structure = Atoms(symbols=symbols, 
                             scaled_positions=scaled_positions, 
                             cell=cell, 
                             pbc=True)

        # Convert to AiiDA StructureData
        structure = StructureData(ase=ase_structure)

        # Relaxation parameters
        wf_relax = Dict(dict={
            'film_distance_relaxation': False,
            'force_criterion': 0.049,   # Convergence threshold (eV/Å)
            'relax_iter': 10           # Maximum relaxation steps
        })
        
        # SCF calculation parameters
        wf_relax_scf = Dict(dict={
            'fleur_runmax': 10,        # Max SCF cycles
            'itmax_per_run': 100,      # Max iterations per SCF
            'energy_converged': 0.0001,# Energy convergence (eV)
            'mode': 'energy',          # Convergence criterion
            'force_dict': {            # Force mixing parameters
                'qfix': 2,
                'forcealpha': 0.5,
                'forcemix': 'straight'
            }
        })
        
        # Computational resource configuration
        options_relax = Dict(dict={
            'resources': {
                'num_machines': 1,
                'num_mpiprocs_per_machine': 1,
                'num_cores_per_mpiproc': 4
            },
            'queue_name': 'devel',
            'max_wallclock_seconds': 10 * 60 * 6000  # 10 hours
        })

        # Submit FleurRelaxWorkChain
        future = self.submit(
            FleurRelaxWorkChain,
            scf={
                'wf_parameters': wf_relax_scf,
                'options': options_relax,
                'inpgen': inpgen_code,
                'fleur': fleur_code,
                'structure': structure
            },
            wf_parameters=wf_relax
        )
        return ToContext(energy_future=future)  # Track calculation status

    def finalize_energy(self):
        """Extract and validate energy from completed calculation.
        
        Handles:
        - Successful energy extraction
        - Missing output parameters
        - Unexpected errors during parsing
        """
        energy_value = None
        
        if self.ctx.energy_future.is_finished_ok:
            try:
                # Retrieve energy from work chain outputs
                output_para = self.ctx.energy_future.outputs.output_relax_wc_para
                total_energy = output_para.get_dict().get('energy')
                
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
        self.out('energy', energy_node)