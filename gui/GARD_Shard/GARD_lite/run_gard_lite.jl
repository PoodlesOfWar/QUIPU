# ===========================================================================
# GARD Lite — Julia High-Performance Neural Tensor Compression Launcher
# ===========================================================================
# Runs the GARD Lite self-sustainable compression/decompression application,
# starts the engine background server on port 8780, and opens gard_lite.html
# directly in the default web browser.
# ===========================================================================

using Sockets
using Dates

const SCRIPT_DIR = @__DIR__
const HTML_PATH = joinpath(SCRIPT_DIR, "gard_lite.html")
const INDEX_PATH = joinpath(SCRIPT_DIR, "index.html")
const ENGINE_PATH = joinpath(SCRIPT_DIR, "gard_lite_engine.py")
const PORT = 8780
const SERVER_URL = "http://127.0.0.1:$PORT/"

function is_server_online(port::Int)::Bool
    try
        s = connect("127.0.0.1", port)
        close(s)
        return true
    catch
        return false
    end
end

function open_browser(target::String)
    if Sys.iswindows()
        run(`cmd /c start "" "$target"`, wait=false)
    elif Sys.isapple()
        run(`open "$target"`, wait=false)
    else
        run(`xdg-open "$target"`, wait=false)
    end
end

function main()
    println("=================================================================")
    println("  GARD Lite — Julia Neural Tensor Control Plane Launcher         ")
    println("=================================================================")
    println("  Time:           ", Dates.now())
    println("  Script Dir:     ", SCRIPT_DIR)
    println("  Server Port:    ", PORT)
    println("  Target URL:     ", SERVER_URL)
    println("=================================================================")

    if is_server_online(PORT)
        println("✔ GARD Lite engine server is already running on port $PORT.")
    else
        println("➜ Starting GARD Lite engine background server...")
        if Sys.iswindows()
            run(`powershell -Command "Start-Process python -ArgumentList '$ENGINE_PATH --port $PORT' -WindowStyle Hidden"`, wait=false)
        else
            run(`python3 "$ENGINE_PATH" --port $PORT`, wait=false)
        end
        sleep(1.5)
    end

    target_to_open = isfile(HTML_PATH) ? HTML_PATH : (isfile(INDEX_PATH) ? INDEX_PATH : SERVER_URL)
    println("➜ Opening GARD Lite HTML Control Plane in browser: $target_to_open")
    open_browser(target_to_open)
    println("✔ Launch sequence complete.")
end

main()
