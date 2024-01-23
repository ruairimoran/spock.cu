NOTE: use `plantuml-svg` for best render.
```plantuml
!theme vibrant

abstract host.python_api {
	+generate_data()
	+matplotlib()
	+tikz()
}
abstract host.cpp_api {
	+NOT USED YET
}

interface host.json.data
interface host.json.result

class host.ScenarioTree {  
	+num_nodes: int
	+num_nonleaf_nodes: int
	+horizon: int
	+ancestors: vec<int>
	+stages: vec<int>
	+get_children_from_of_node()
	+get_children_to_of_node()
}

class host.ProblemData {  
   +A: vec<int>
   +B: vec<int>
   +L: vec<int>
}

class host.Cache {
	+prim: vec<int>
	+dual: vec<int>
}

interface host.HostMemory {
	+Cache_h: ptr
}

interface device.DeviceMemory {
	+Cache_d: ptr
}

class device.Engine {
	+solution: vec<int>
}

host.python_api --> host.json.data
host.cpp_api --> host.ScenarioTree
host.cpp_api --> host.ProblemData

host.json.data --> host.ScenarioTree
host.json.data --> host.ProblemData

host.ScenarioTree --> host.Cache
host.ProblemData --> host.Cache

host.Cache <--> host.HostMemory
host.HostMemory <--> device.DeviceMemory
device.DeviceMemory <--> device.Engine

host.Cache --> host.json.result : "testing"
host.json.result --> host.python_api : "testing"

@enduml
```