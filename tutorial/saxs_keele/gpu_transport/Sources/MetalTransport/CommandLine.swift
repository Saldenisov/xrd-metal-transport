import Foundation

@main
enum MetalTransportCLI {
    static func main() {
        do {
            let arguments = CommandLine.arguments
            if arguments.count > 1 && arguments[1] == "--list-devices" {
                try listMetalDevices()
            } else if arguments.count > 1 && arguments[1] == "--probe-compton" {
                try runComptonProbe(arguments: arguments)
            } else {
                try runTransport(arguments: arguments)
            }
        } catch {
            fputs("\(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }
}
