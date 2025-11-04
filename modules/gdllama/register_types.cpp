
#include "register_types.h"
#include "core/object/class_db.h"

#include "gdllama.h"
#include "gdllama_rag.h"
#include "gdllama_diffusion.h"
#include "gdllama_vision.h"

void initialize_gdllama_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
			return;
	}
	ClassDB::register_class<Llama>();
	ClassDB::register_class<Rag>();
	ClassDB::register_class<Diffusion>();
	ClassDB::register_class<Vision>();
}

void uninitialize_gdllama_module(ModuleInitializationLevel p_level) {

}
