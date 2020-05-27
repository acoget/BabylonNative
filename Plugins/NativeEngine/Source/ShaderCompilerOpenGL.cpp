#include "ShaderCompiler.h"
#include "ResourceLimits.h"
#include <arcana/experimental/array.h>
#include <glslang/Public/ShaderLang.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv_parser.hpp>
#include <spirv_glsl.hpp>

namespace Babylon
{
    extern const TBuiltInResource DefaultTBuiltInResource;

    namespace
    {
        void AddShader(glslang::TProgram& program, glslang::TShader& shader, std::string_view source)
        {
            const std::array<const char*, 1> sources{source.data()};
            shader.setStrings(sources.data(), gsl::narrow_cast<int>(sources.size()));
            shader.setEnvInput(glslang::EShSourceGlsl, shader.getStage(), glslang::EShClientVulkan, 100);
            shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
            shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

            if (!shader.parse(&DefaultTBuiltInResource, 450, false, EShMsgDefault))
            {
                throw std::exception();
            }

            program.addShader(&shader);
        }

        std::unique_ptr<spirv_cross::Compiler> CompileShader(glslang::TProgram& program, EShLanguage stage, std::string& glsl)
        {
            std::vector<uint32_t> spirv;
            glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);

            spirv_cross::Parser parser{std::move(spirv)};
            parser.parse();

            auto compiler = std::make_unique<spirv_cross::CompilerGLSL>(parser.get_parsed_ir());

            spirv_cross::CompilerGLSL::Options options = compiler->get_common_options();

#ifdef ANDROID
            options.version = 300;
            options.es = true;
#else
            options.version = 430;
            options.es = false;
#endif
            // This will make the struct emit as struct declaration + a variable, which is easier to work around. See below.
            options.emit_uniform_buffer_as_plain_uniforms = true;

            compiler->set_common_options(options);

            // glslang works with Vulkan GLSL, which requires separate textures and samplers.
            // But GL wants combined samplers + textures, so we build those.
            compiler->build_combined_image_samplers();

            // Remap the combined sampler names to human-friendly names and re-add the lost binding slot.
            // Doing this here means the names and binding will be correct in the shader source, and we can just use the separate samplers to build the bgfx header.
            const spirv_cross::ShaderResources resources = compiler->get_shader_resources();
            auto combinedSamplers = compiler->get_combined_image_samplers();
            for (auto separate : resources.separate_samplers)
            {
                auto id = separate.id;
                auto& samplerName = separate.name;
                auto binding = compiler->get_decoration(id, spv::DecorationBinding);
                for (auto combined : combinedSamplers)
                {
                    if (combined.sampler_id == id)
                    {
                        id = combined.combined_id;
                        break;
                    }
                }
                compiler->set_name(id, samplerName);
                compiler->set_decoration(id, spv::DecorationBinding, binding);
            }

            // SPIRV-Cross will only ever output uniform structs, whether as UBOs or as plain uniforms.
            // That causes the names to be struct.field in GL reflection, which is what bgfx uses.
            // If the structs have different names between the vertex and fragment stages, the same uniform gets duplicated due to the differece in qualified name, and only set once.
            // If the structs have the same name between the vertex and fragment stages, but aren't identical (most cases), GLSL can't link the program.
            // Therefore, the only feasible approach is to modify the shader code output by SPIRV-Cross and put plain non-struct uniforms back in.
            // This cannot be done from Javascript, where it would be trivial, because value type uniforms that aren't inside a struct are invalid in Vulkan GLSL, which is what we use.
            // And we can't use GL GLSL, because then neither the uniforms nor the samplers show up in the SPIRV-Cross reflection data we need to build the bgfx shader header.
            // In practice, this adds a second copy of the uniforms in the header string before emitting the code.
            // The struct will remain in shader code, unused, but it was the source of the reflection info that both this and NativeEngine use.
            const spirv_cross::Resource uniformBuffer = resources.uniform_buffers[0];
            const spirv_cross::SPIRType& type = compiler->get_type(uniformBuffer.base_type_id);
            for (uint32_t index = 0; index < type.member_types.size(); ++index)
            {
                auto memberType = compiler->get_type(type.member_types[index]);

                std::string uniformType;
                auto uniformName = compiler->get_member_name(uniformBuffer.base_type_id, index);

                // JS should have taken care of turning all uniforms into either vec4 or mat4
                if (memberType.columns == 1 && memberType.vecsize == 4)
                    uniformType = "vec4 ";
                else if (memberType.columns == 4 && memberType.vecsize == 4)
                    uniformType = "mat4 ";
                else
                    throw std::exception();

                compiler->add_header_line("uniform " + uniformType + uniformName + ";");
            }

            // Rename the struct and make it clear it's unused
            compiler->set_name(uniformBuffer.id, stage == EShLangVertex ? "UnusedVS" : "UnusedFS");
            compiler->unset_decoration(uniformBuffer.id, spv::DecorationBinding);

            std::string compiled = compiler->compile();

            // Since we know exactly how the unused struct instance looks, we can comment it out to avoid noise in GL reflection...
            const std::string unusedUniform = "uniform Frame Unused";
            size_t pos = compiled.find(unusedUniform);
            if (pos != std::string::npos)
            {
                compiled.replace(pos, unusedUniform.size(), "//uniform Frame Unused");
            }
            // ... and rewrite all the accesses so the shader actually compiles.
            const std::string unusedUniformAccess = stage == EShLangVertex ? "UnusedVS." : "UnusedFS.";
            pos = compiled.find(unusedUniformAccess);
            while (pos != std::string::npos)
            {
                compiled.replace(pos, unusedUniformAccess.size(), "");
                pos = compiled.find(unusedUniformAccess);
            }

#ifdef ANDROID
            glsl = compiled.substr(strlen("#version 300 es\n"));

            // frag def
            static const std::string fragDef = "layout(location = 0) out highp vec4 glFragColor;";
            pos = glsl.find(fragDef);
            if (pos != std::string::npos)
            {
                glsl.replace(pos, fragDef.size(), "");
            }

            // frag
            static const std::string fragColor = "glFragColor";
            pos = glsl.find(fragColor);
            if (pos != std::string::npos)
            {
                glsl.replace(pos, fragColor.size(), "gl_FragColor");
            }
#else
            glsl = compiled;
#endif
            return std::move(compiler);
        }
    }

    ShaderCompiler::ShaderCompiler()
    {
        glslang::InitializeProcess();
    }

    ShaderCompiler::~ShaderCompiler()
    {
        glslang::FinalizeProcess();
    }

    void ShaderCompiler::Compile(std::string_view vertexSource, std::string_view fragmentSource, std::function<void(ShaderInfo, ShaderInfo)> onCompiled)
    {
        glslang::TProgram program;

        glslang::TShader vertexShader{EShLangVertex};
        AddShader(program, vertexShader, vertexSource);

        glslang::TShader fragmentShader{EShLangFragment};
        AddShader(program, fragmentShader, fragmentSource);

        if (!program.link(EShMsgDefault))
        {
            throw std::exception();
        }

        std::string vertexGLSL(vertexSource.data(), vertexSource.size());
        auto vertexCompiler = CompileShader(program, EShLangVertex, vertexGLSL);

        std::string fragmentGLSL(fragmentSource.data(), fragmentSource.size());
        auto fragmentCompiler = CompileShader(program, EShLangFragment, fragmentGLSL);

        uint8_t* strVertex = (uint8_t*)vertexGLSL.data();
        uint8_t* strFragment = (uint8_t*)fragmentGLSL.data();
        onCompiled(
            {std::move(vertexCompiler), gsl::make_span(strVertex, vertexGLSL.size())},
            {std::move(fragmentCompiler), gsl::make_span(strFragment, fragmentGLSL.size())});
    }
}
